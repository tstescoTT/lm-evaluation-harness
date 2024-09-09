import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import Collator, pad_and_concat, stop_sequences_criteria


@register_model("hf-multimodal")
class HFMultimodalLM(HFLM):
    """
    An abstracted Hugging Face model class for multimodal LMs like Llava and Idefics.
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForVision2Seq  # TODO: what's the right way to handle this. maybe phase out the direct class-equality checks in HFLM?

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.ProcessorMixin,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Helper method during initialization.

        For the multimodal variant, we initialize not just
        `self.tokenizer` but also `self.processor`.
        """

        if tokenizer:
            if isinstance(tokenizer, str):
                return transformers.AutoProcessor.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    # use_fast=use_fast_tokenizer,
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                return tokenizer

        # Get tokenizer based on 'pretrained'
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            # get the HF hub name via accessor on model
            model_name = self.model.name_or_path

        self.processor = transformers.AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            # use_fast=use_fast_tokenizer,
        )

        self.tokenizer = self.processor.tokenizer

    # def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
    #     """
    #     Method to apply a chat template to a list of chat history between user and model.
    #     """
    #     return self.tokenizer.apply_chat_template(
    #         chat_history, tokenize=False, add_generation_prompt=True
    #     )

    # def tok_encode(
    #     self, string: str, left_truncate_len=None, add_special_tokens=None
    # ) -> List[int]:
    #     """ """
    #     # default for None - empty dict, use predefined tokenizer param
    #     # used for all models except for CausalLM or predefined value
    #     special_tokens_kwargs = {}

    #     # by default for CausalLM - false or self.add_bos_token is set
    #     if add_special_tokens is None:
    #         if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
    #             special_tokens_kwargs = {
    #                 "add_special_tokens": False or self.add_bos_token
    #             }
    #     # otherwise the method explicitly defines the value
    #     else:
    #         special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

    #     encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

    #     # left-truncate the encoded context to be at most `left_truncate_len` tokens long
    #     if left_truncate_len:
    #         encoding = encoding[-left_truncate_len:]

    #     return encoding

    def tok_batch_encode(
        self,
        strings: List[str],  # note that input signature of this fn is different
        visuals=None,  # TODO: typehint on this
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Dict[
        str, torch.Tensor
    ]:  # TODO: note that this return signature differs from HFLM tok_batch_encode.
        # TODO: we should allow

        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.processor(
            images=visuals,
            text=strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        # if "pixel_values" in encoding and encoding["pixel_values"] is None: # TODO: what case does this handle?
        #     encoding.pop("pixel_values")

        encoding.to(
            self.device, self.model.dtype
        )  # TODO: casting to dtype seems odd for input_ids and attn_mask.
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding

    # def tok_decode(self, tokens, skip_special_tokens=True):
    #     return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, imgs, attn_mask=None, labels=None):
        """
        TODO: update docstring
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            return self.model(inps, imgs).logits

    def _model_generate(self, inputs, max_length, stop, **generation_kwargs):
        # gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer,
            stop,
            inputs["input_ids"].shape[1],
            inputs["input_ids"].shape[0],
        )
        return self.model.generate(
            **inputs,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "model type `hf-multimodal` does not support loglikelihood_rolling. Use 'hf' model type for text-only loglikelihood_rolling tasks ",
            "this is because we do not support measuring the loglikelihood a model assigns to an image.",
        )

    def loglikelihood(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation, aux_arguments in [req.args for req in requests]:
            if context == "":
                raise ValueError("Must get non-empty context for multimodal requests!")
            else:
                visuals = aux_arguments["visual"]

                context_enc, continuation_enc = self._encode_pair(context, continuation)
            # TODO: key to pick for caching images
            new_reqs.append(
                (
                    (visuals, context, continuation),
                    visuals,
                    context_enc,
                    continuation_enc,
                )
            )

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(
        self,
        requests: List[
            Tuple[Tuple[None, str, str], List[int], List[int], List[int]]
        ],  # TODO: update typehint to be correct
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[2] + req[3]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-3] + req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"  # TODO: can't group-by just "contexts" any more, need to incorporate imgs
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
            and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            imgs = []
            inps = []
            cont_toks_list = []
            inplens = []

            padding_len_inp = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, image_enc, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(image_enc) > 0
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                # TODO: assuming that we won't handle enc-dec Vision2Seq models. Is that a safe assumption?
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

                # image encoding
                image_enc = self.processor(  # TODO: factor this call into an image_encode method??
                    text="<image>"
                    * len(
                        image_enc
                    ),  # we need this to process images for some models, annoyingly
                    images=image_enc,
                    return_tensors="pt",  # TODO: push tensor conversion into _loglikelihood_tokens
                    # **add_special_tokens,
                )[
                    "pixel_values"
                ]  # TODO: are keys from processors always the same? just pop attention_mask and input_ids keys

                imgs.append(image_enc)

                # TODO: batch input images together...

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            batched_inps = pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]
            batched_imgs = torch.cat(imgs, dim=0)  # TODO: might be wrong, fix

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, batched_imgs, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            for (
                request_str,
                image_encs,
                ctx_tokens,
                _,
            ), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        # TODO: back out to HFLM.generate_until() for all requests without aux_arguments
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests with text+image input",
        )
        # TODO: port auto-batch sizing into this.

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(
            [reg.args for reg in requests],
            _collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        ### Up to here: was identical to non-multimodal HFLM generate_until ###

        for chunk in chunks:
            contexts, all_gen_kwargs, aux_arguments = zip(
                *chunk
            )  # TODO: can we cut down further on number of distinct things we pass around?

            visuals = [
                arg["visual"] for arg in aux_arguments
            ]  # TODO: I think *fully* flattening is just wrong for bs>1 ??

            if not isinstance(contexts, list):
                contexts = list(
                    contexts
                )  # for Qwen2-VL, processor is unhappy accepting a tuple of strings instead of a list.
                # TODO: could we upstream this workaround?
            ### this part onward: same as HFLM ###

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            ### end stuff that's entirely copied verbatim from HFLM ###

            max_ctx_len = self.max_length - max_gen_toks  # noqa: F841 # TODO: this assumes we are using a causal LM. is that always valid? shouldn't be

            print(visuals, contexts)
            inputs = self.tok_batch_encode(
                contexts,
                visuals,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            ).to(self.device, self.model.dtype)

            context_enc = inputs["input_ids"]

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            cont = self._model_generate(inputs, stop=until, **kwargs)

            del inputs
            torch.cuda.empty_cache()
            import gc

            gc.collect()

            ### essentially same as HFLM beyond this line!

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                # if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM: # TODO: ensure this holds for VLMs
                cont_toks = cont_toks[context_enc.shape[1] :]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), s
                )  # TODO: cache key for multimodal input should be what?
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
