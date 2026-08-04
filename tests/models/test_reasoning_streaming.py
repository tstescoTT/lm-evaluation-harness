"""Reasoning-content capture across the chat streaming + logging path.

These cover the additions that let reasoning-capable chat endpoints carry a
separate ``reasoning_content`` trace through evaluation without letting it leak
into the scored answer:

- SSE streaming (async + sync) accumulates content and reasoning separately.
- ``LocalChatCompletion.parse_generations`` wraps the answer in
  ``ChatGeneration`` so the reasoning rides along as an attribute.
- ``EvaluationTracker`` extracts an aligned ``reasoning_content`` field before
  ``sanitize_list`` collapses the string subclass.
"""

import asyncio
from types import SimpleNamespace

from lm_eval.loggers.evaluation_tracker import _extract_reasoning
from lm_eval.models.api_models import (
    ChatGeneration,
    TemplateAPI,
    _consume_requests_sse_stream,
)
from lm_eval.models.openai_completions import LocalChatCompletion


class _AsyncContent:
    def __init__(self, lines):
        self._lines = iter(lines)

    async def readline(self):
        return next(self._lines, b"")


_CHAT_STREAM_LINES = [
    b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"think "}}]}\n',
    b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"carefully"}}]}\n',
    b'data: {"choices":[{"index":0,"delta":{"content":"42"}}]}\n',
    b"data: [DONE]\n",
]


def test_async_stream_message_carries_reasoning_separately():
    response = SimpleNamespace(content=_AsyncContent(_CHAT_STREAM_LINES))

    result = asyncio.run(TemplateAPI._consume_sse_stream(None, response))

    message = result["choices"][0]["message"]
    assert message["content"] == "42"
    assert message["reasoning_content"] == "think carefully"


def test_sync_stream_message_carries_reasoning_separately():
    def iter_lines(decode_unicode=True):
        for line in _CHAT_STREAM_LINES:
            yield line.decode("utf-8")

    response = SimpleNamespace(iter_lines=iter_lines)

    result = _consume_requests_sse_stream(response)
    message = result["choices"][0]["message"]
    assert message["content"] == "42"
    assert message["reasoning_content"] == "think carefully"


def test_streamed_parse_wraps_reasoning_when_enabled(monkeypatch):
    monkeypatch.setenv("LM_EVAL_PRESERVE_REASONING", "1")
    response = SimpleNamespace(content=_AsyncContent(_CHAT_STREAM_LINES))
    result = asyncio.run(TemplateAPI._consume_sse_stream(None, response))

    # The chat parser exposes only the answer, with reasoning on the attribute.
    (parsed,) = LocalChatCompletion.parse_generations(result)
    assert parsed == "42"
    assert isinstance(parsed, ChatGeneration)
    assert parsed.reasoning_content == "think carefully"


def test_text_completions_stream_shape_unchanged():
    lines = [
        b'data: {"choices":[{"index":0,"text":"hello "}]}\n',
        b'data: {"choices":[{"index":0,"text":"world"}]}\n',
        b"data: [DONE]\n",
    ]
    response = SimpleNamespace(content=_AsyncContent(lines))

    result = asyncio.run(TemplateAPI._consume_sse_stream(None, response))

    assert result == {"choices": [{"index": 0, "text": "hello world"}]}


def test_non_streaming_parse_wraps_reasoning_when_enabled(monkeypatch):
    monkeypatch.setenv("LM_EVAL_PRESERVE_REASONING", "1")
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": "final answer",
                    "reasoning_content": "private work",
                },
            }
        ]
    }

    (parsed,) = LocalChatCompletion.parse_generations(response)
    assert parsed == "final answer"
    assert isinstance(parsed, ChatGeneration)
    assert parsed.reasoning_content == "private work"


def test_parse_is_plain_str_when_reasoning_capture_disabled(monkeypatch):
    # Default (env unset): reasoning present but not requested -> plain answer.
    monkeypatch.delenv("LM_EVAL_PRESERVE_REASONING", raising=False)
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": "final answer",
                    "reasoning_content": "private work",
                },
            }
        ]
    }

    (parsed,) = LocalChatCompletion.parse_generations(response)
    assert parsed == "final answer"
    assert not isinstance(parsed, ChatGeneration)


def test_non_streaming_parse_without_reasoning_is_plain_str(monkeypatch):
    monkeypatch.setenv("LM_EVAL_PRESERVE_REASONING", "1")
    response = {"choices": [{"index": 0, "message": {"content": "plain"}}]}

    (parsed,) = LocalChatCompletion.parse_generations(response)
    assert parsed == "plain"
    assert not isinstance(parsed, ChatGeneration)


def test_extract_reasoning_aligns_with_resps_structure():
    resps = [[ChatGeneration("answer", "reasoning")]]

    found, reasoning = _extract_reasoning(resps)

    assert found is True
    assert reasoning == [["reasoning"]]


def test_extract_reasoning_absent_when_no_generation_carries_it():
    found, reasoning = _extract_reasoning([["plain answer"]])

    assert found is False
    assert reasoning == [[None]]
