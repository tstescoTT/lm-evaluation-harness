import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI, JsonChatStr
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import eval_logger
from lm_eval.api.instance import Instance


@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="huggingface",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=itemgetter("index")), ctxlens
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["text"]
            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")


@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        eval_logger.warning(
            "chat-completions endpoint requires the `--apply_chat_template` flag."
        )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        # assert (
        #     type(messages) is not str
        # ), "chat-completions require the --apply_chat_template flag."
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["message"]["content"]
            res = res + tmp
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions. Consider using the completions API instead."
        )


@register_model(
    "openai-completions",
)
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert (
            self.model
            in [
                "babbage-002",
                "davinci-002",
            ]
        ), f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert (
            type(messages) is not str
        ), "chat-completions require the --apply_chat_template flag."
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if "o1" in self.model:
            output.pop("stop")
            output["temperature"] = 1
        return output


DEFAULT_IMAGE_PLACEHOLDER = "<image>"

@register_model("local-mm-chat-completions")
class LocalMMChatCompletion(LocalCompletionsAPI):
    """Local Chat Completions API implementation with multimodal support."""
    MULTIMODAL = True

    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=False,
        max_images=999,  # Matches VLLM_VLM default
        **kwargs,
    ):
        eval_logger.warning(
            "chat-completions endpoint requires the `--apply_chat_template` flag."
        )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        
        self.max_images = max_images
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def create_message(
        self,
        messages: Union[List[Tuple[str, Dict[str, Any]]], List[str], List[JsonChatStr]],
        generate: bool = False
    ) -> List[Dict[str, Any]]:
        """Format raw messages into the proper chat format.
        
        Args:
            messages: Raw messages, can be text-only or (text, visuals) pairs
            generate: Whether this is a generation request
            
        Returns:
            List of formatted message dictionaries
        """
        formatted_messages = []
        
        # Handle different message formats
        for msg in messages:
            if isinstance(msg, tuple):
                # Handle (context, visuals) pair
                context, visual_data = msg
                content = []
                
                # Add text content
                content.append({
                    "type": "text",
                    "text": context
                })
                
                # Add image content if present
                if visual_data and "visual" in visual_data:
                    images = visual_data["visual"]
                    if not isinstance(images, list):
                        images = [images]
                        
                    # Limit number of images
                    images = images[:self.max_images]
                    
                    for image in images:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image)}"
                            }
                        })
                
                formatted_messages.append({
                    "role": "user", 
                    "content": content
                })
            elif isinstance(msg, JsonChatStr):
                # Handle JSON-encoded chat history
                formatted_messages.extend(json.loads(msg.prompt))
            else:
                # Handle plain text message
                formatted_messages.append({
                    "role": "user",
                    "content": str(msg)
                })
                
        return formatted_messages

    def _create_payload(
        self,
        messages: List[Dict[str, Any]],
        generate: bool = False,
        gen_kwargs: dict = None,
        seed: int = 1234,
        eos: str = None,
        **kwargs,
    ) -> dict:
        """Create the API payload with support for multimodal inputs.
        
        Args:
            messages: Pre-formatted messages from create_message()
            generate: Whether this is a generation request
            gen_kwargs: Generation parameters
            seed: Random seed
            eos: End of sequence token
            
        Returns:
            Formatted API payload
        """
        assert messages, "No messages provided"

        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        
        # Handle max tokens
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            
        # Handle temperature
        temperature = gen_kwargs.pop("temperature", 0)
        
        # Handle stop sequences
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        # Create the payload
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            **gen_kwargs,
        }
        
        # Add stop sequences if present (max 4 for OpenAI API)
        if stop and stop[0] is not None:
            payload["stop"] = stop[:4]
            
        return payload
        
        # Add stop sequences (max 4 as per OpenAI API)
        if stop and stop[0] is not None:
            payload["stop"] = stop[:4]

        return payload

    def generate_until(
        self,
        requests: List[Instance],
        disable_tqdm: bool = False
    ) -> List[str]:
        """Generate responses for text and image inputs.
        
        Args:
            requests: List of Instance objects containing generation requests
            disable_tqdm: Whether to disable the progress bar
            
        Returns:
            List of generated text responses
        """
        res = []
        
        def _collate_gen(_requests):
            # Sort by length of contexts
            return -len(_requests[0])
        
        # Extract request arguments
        request_args = []
        for req in requests:
            # Each req.args should be (context, gen_kwargs, visuals)
            context, gen_kwargs, visuals = req.args
            # Create tuple of context and visuals for the message formatting
            request_args.append((context, gen_kwargs, visuals))
            
        # Create batches grouped by gen_kwargs
        re_ord = Collator(
            request_args,
            sort_fn=_collate_gen,
            group_by="gen_kwargs"
        )
        chunks = re_ord.get_batched(n=self._batch_size)

        # Process each batch
        pbar = tqdm(
            desc="Requesting API",
            total=len(requests),
            disable=disable_tqdm
        )
        
        for chunk in chunks:
            # Unpack the chunks
            contexts, gen_kwargs, visuals = zip(*chunk)
            
            # Format messages with context and visuals
            messages = list(zip(contexts, visuals))
            
            # Call API
            outputs = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                reraise=True
            )(self.model_call)(
                messages=messages,
                generate=True,
                gen_kwargs=copy.deepcopy(gen_kwargs[0])  # All kwargs in batch are same
            )

            # Process outputs
            for generated_text, context in zip(
                self.parse_generations(outputs=outputs),
                contexts
            ):
                if generated_text is not None:
                    res.append(generated_text)
                    # Cache the result
                    self.cache_hook.add_partial(
                        "generate_until",
                        (context, gen_kwargs[0]),
                        generated_text
                    )
                    pbar.update(1)

        pbar.close()
        
        # Restore original order
        return re_ord.get_original(res)

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """Parse generation outputs from the API response."""
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
            
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                # Extract the text content from the message
                if isinstance(choices["message"]["content"], list):
                    # For multimodal responses, concatenate text contents
                    text_contents = [
                        content["text"] 
                        for content in choices["message"]["content"] 
                        if content["type"] == "text"
                    ]
                    tmp[choices["index"]] = " ".join(text_contents)
                else:
                    tmp[choices["index"]] = choices["message"]["content"]
            res.extend(tmp)
            
        return res

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions with images. "
            "Consider using the completions API instead."
        )

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        """Return empty string as chat template is handled by the API."""
        return ""
