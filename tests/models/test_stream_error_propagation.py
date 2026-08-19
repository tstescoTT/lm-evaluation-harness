"""In-stream API errors must surface, not silently shrink the response list.

A streaming endpoint can answer ``200 OK`` and only then report the failure
inside the stream body::

    data: {"error":{"message":"...max session count reached...","code":500}}
    data: [DONE]

Such a chunk carries no ``choices``, so the stream consumers used to return an
empty ``choices`` list. ``parse_generations`` then produced zero answers for a
request that was counted as successful, the response list came back shorter than
the request list, and the run died far away in ``Collator.get_original`` with
``ValueError: zip() argument 2 is shorter than argument 1`` -- with the server's
message nowhere in the logs.

These cover both halves of the fix: the consumers raise (so tenacity retries and
the message is logged), and ``get_batched_requests`` enforces one response per
request no matter what a batch returns.
"""

import asyncio
from types import SimpleNamespace

import pytest

from lm_eval.models.api_models import (
    TemplateAPI,
    _consume_requests_sse_stream,
    _sse_error_message,
)


_ERROR_CHUNK = (
    b'data: {"error":{"message":"{\\"code\\":500,\\"message\\":\\"session '
    b"resolution failed: Failed to allocate: max session count reached after "
    b'all attempts\\"}","type":"internal_server_error","code":500}}\n'
)
_DONE = b"data: [DONE]\n"


class _AsyncContent:
    def __init__(self, lines):
        self._lines = iter(lines)

    async def readline(self):
        return next(self._lines, b"")


def _sync_response(lines):
    def iter_lines(decode_unicode=True):
        for line in lines:
            yield line.decode("utf-8")

    return SimpleNamespace(iter_lines=iter_lines)


@pytest.mark.parametrize(
    "chunk, expected",
    [
        ({"error": {"message": "boom"}}, "boom"),
        ({"error": "boom"}, "boom"),
        ({"choices": [{"index": 0, "delta": {"content": "hi"}}]}, None),
        ({"choices": []}, None),
        ({}, None),
    ],
)
def test_sse_error_message_only_fires_on_error_payloads(chunk, expected):
    assert _sse_error_message(chunk) == expected


def test_async_stream_raises_with_server_message():
    response = SimpleNamespace(content=_AsyncContent([_ERROR_CHUNK, _DONE]))

    with pytest.raises(RuntimeError, match="max session count reached"):
        asyncio.run(TemplateAPI._consume_sse_stream(None, response))


def test_sync_stream_raises_with_server_message():
    with pytest.raises(RuntimeError, match="max session count reached"):
        _consume_requests_sse_stream(_sync_response([_ERROR_CHUNK, _DONE]))


def test_error_after_tokens_keeps_them_as_partial_output():
    """Tokens already streamed are worth more than a retry from scratch."""
    lines = [
        b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n',
        _ERROR_CHUNK,
        _DONE,
    ]

    result = _consume_requests_sse_stream(_sync_response(lines))

    content = result["choices"][0]["message"]["content"]
    assert "hel" in content
    assert "__PARTIAL_OUTPUT__" in content
    assert "max session count reached" in content


def test_clean_stream_is_unaffected():
    lines = [
        b'data: {"choices":[{"index":0,"delta":{"content":"4"}}]}\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"2"}}]}\n',
        _DONE,
    ]

    result = _consume_requests_sse_stream(_sync_response(lines))

    assert result["choices"][0]["message"]["content"] == "42"


class _BatchAPI(TemplateAPI):
    """Returns a canned per-call result, bypassing all network setup."""

    def __init__(self, results):
        self._results = list(results)
        self._concurrent = len(self._results)
        self._batch_size = 1
        self.max_retries = 1
        self.timeout = 5
        self.verify_certificate = False
        self.base_url = "http://localhost"
        self._seed = 1234
        self.tokenized_requests = False
        self._calls = 0

    async def amodel_call(self, session, sem, messages, **kwargs):
        result = self._results[self._calls]
        self._calls += 1
        if isinstance(result, BaseException):
            raise result
        return result

    def _create_payload(self, *args, **kwargs):
        return {}

    @staticmethod
    def parse_generations(outputs, **kwargs):
        return []

    @staticmethod
    def parse_logprobs(outputs, **kwargs):
        return []

    @property
    def header(self):
        return {}


def _run_batches(results, generate=True):
    api = _BatchAPI(results)
    requests = [f"req{i}" for i in range(len(results))]
    batches = asyncio.run(
        api.get_batched_requests(
            requests, cache_keys=[None] * len(requests), generate=generate
        )
    )
    return [item for batch in batches for item in batch]


def test_short_batch_is_padded_so_lengths_stay_aligned():
    flat = _run_batches(
        [
            ["good answer"],
            [],  # the bug: a stream that yielded no choices
            RuntimeError("session resolution failed: max session count reached"),
        ]
    )

    assert len(flat) == 3
    assert flat[0] == "good answer"
    assert flat[1].startswith("__INFERENCE_ERROR__")
    assert "0 response(s) for a batch of 1" in flat[1]
    assert "max session count reached" in flat[2]


def test_short_batch_padding_for_loglikelihood():
    flat = _run_batches([[(-1.5, True)], []], generate=False)

    assert len(flat) == 2
    assert flat[0] == (-1.5, True)
    assert flat[1] == (float("-inf"), False)


def test_blocking_path_keeps_one_answer_per_request():
    """The concurrency<=1 path zips non-strictly, so it needs the same guard."""

    class _ShortParseAPI(_BatchAPI):
        def __init__(self):
            super().__init__([])
            self._concurrent = 1
            self._batch_size = 2
            self._max_gen_toks = 8

        def model_call(self, messages, **kwargs):
            return {"choices": []}

        @staticmethod
        def parse_generations(outputs, contexts=None, **kwargs):
            return ["only one answer"]  # caller asked for two

        def apply_chat_template(self, chat_history, **kwargs):
            return chat_history

    api = _ShortParseAPI()
    api.cache_hook = SimpleNamespace(add_partial=lambda *args, **kwargs: None)
    requests = [
        SimpleNamespace(args=("ctx a", {"until": ["\n"]})),
        SimpleNamespace(args=("ctx b", {"until": ["\n"]})),
    ]

    res = api.generate_until(requests)

    assert len(res) == 2
    assert res[0] == "only one answer"
    assert res[1].startswith("__INFERENCE_ERROR__")
    assert "1 response(s) for a batch of 2" in res[1]
