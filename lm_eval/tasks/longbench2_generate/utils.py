# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Generate-then-answer LongBench v2 loader, answer extraction, and scoring.

Stock lm-eval longbench2 is multiple_choice (loglikelihood), which does not
work with chat-completions-only servers. This generate variant asks the model
to produce an answer (optionally with reasoning) and extracts A/B/C/D.

The loader additionally supports filtering the dataset by input sequence
length (ISL). ISL is measured by tokenizing each sample's ``context`` with a
HF tokenizer and keeping only rows whose token count falls within
``[minimum_isl, maximum_isl]``. Both bounds are optional and are supplied from
``eval_config.py`` via ``custom_dataset_kwargs`` (forwarded to lm-eval as
``--metadata`` JSON, then passed here as kwargs).
"""

from __future__ import annotations

import logging
import os
import re
from functools import cache
from typing import Dict, List, Optional

import datasets
from transformers import AutoTokenizer


eval_logger = logging.getLogger(__name__)

_DATASET_PATH = "recursal/longbench-v2"

# All recursal/longbench-v2 subsets (config names). The task concatenates every
# subset into a single dataset and filters by ISL, so a narrow ISL window that
# empties individual domains does not break the run.
_ALL_CONFIGS = (
    "academic_multi",
    "academic_single",
    "agent_history_qa",
    "code_repo_qa",
    "detective",
    "dialogue_history_qa",
    "event_ordering",
    "financial_multi",
    "financial_single",
    "government_multi",
    "government_single",
    "graph_reasoning",
    "legal_multi",
    "legal_single",
    "literary",
    "manyshot_learning",
    "multinews",
    "new_language_translation",
    "table_qa",
    "user_guide_qa",
)


# ---------------------------------------------------------------------------
# Dataset loading + ISL filtering
# ---------------------------------------------------------------------------


@cache
def get_tokenizer(pretrained: Optional[str] = None):
    """Return a cached tokenizer per process (cheap under datasets num_proc)."""
    assert pretrained, "No pretrained tokenizer provided for ISL filtering."
    eval_logger.info(f"Using tokenizer {pretrained} for LongBench v2 ISL filtering.")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def _compute_isl(batch: dict, pretrained: Optional[str] = None) -> dict:
    """Batched map fn: token length of each row's context (input seq length)."""
    tokenizer = get_tokenizer(pretrained=pretrained)
    encoded = tokenizer(batch["context"], add_special_tokens=False)
    return {"_isl": [len(ids) for ids in encoded["input_ids"]]}


def _load_all_subsets() -> datasets.Dataset:
    """Concatenate every recursal/longbench-v2 subset into one dataset."""
    parts = [
        datasets.load_dataset(_DATASET_PATH, cfg, split="train", trust_remote_code=True)
        for cfg in _ALL_CONFIGS
    ]
    return datasets.concatenate_datasets(parts)


def load_longbench2(
    name: Optional[str] = None,
    minimum_isl: Optional[int] = None,
    maximum_isl: Optional[int] = None,
    pretrained: Optional[str] = None,
    tokenizer_num_proc: int = 32,
    **kwargs,
) -> Dict[str, datasets.Dataset]:
    """Load LongBench v2 as a single combined dataset, filtered by ISL range.

    All 20 recursal/longbench-v2 subsets are concatenated into one dataset (so a
    narrow ISL window that empties individual domains does not crash lm-eval),
    then filtered by input sequence length.

    Args:
        name: optional single subset to load instead of all (e.g. ``code_repo_qa``).
        minimum_isl: keep samples whose context tokenizes to >= this many tokens.
        maximum_isl: keep samples whose context tokenizes to <= this many tokens.
        pretrained: HF tokenizer used to measure ISL. Required when filtering.
        tokenizer_num_proc: parallel worker processes for the tokenization pass.
    """
    if name:
        ds = datasets.load_dataset(
            _DATASET_PATH, name, split="train", trust_remote_code=True
        )
    else:
        ds = _load_all_subsets()

    if minimum_isl is None and maximum_isl is None:
        return {"train": ds}

    lo = int(minimum_isl) if minimum_isl is not None else 0
    hi = int(maximum_isl) if maximum_isl is not None else None

    # Disable intra-tokenizer thread parallelism to avoid the fork+threads
    # deadlock/warning once we spawn num_proc worker processes below.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    num_proc = max(1, min(int(tokenizer_num_proc or 1), len(ds)))
    ds = ds.map(
        _compute_isl,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"pretrained": pretrained},
        desc=f"Measuring ISL for longbench2 {name or 'all'}",
    )

    before = len(ds)
    ds = ds.filter(
        lambda isl: (isl >= lo) and (hi is None or isl <= hi),
        input_columns="_isl",
    )
    eval_logger.info(
        f"longbench2 {name or 'all'}: kept {len(ds)}/{before} samples "
        f"with ISL in [{lo}, {hi if hi is not None else 'inf'}]"
    )
    ds = ds.remove_columns("_isl")
    return {"train": ds}


# ---------------------------------------------------------------------------
# Answer extraction + scoring
# ---------------------------------------------------------------------------


def _strip_think(text: str) -> str:
    """Drop reasoning blocks if the model emits them inline."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r".*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_choice(pred: str) -> str:
    """Extract the last A/B/C/D choice from a free-form generation."""
    if not pred:
        return ""

    text = _strip_think(pred)

    # Prefer \boxed{A} / \boxed{(A)}
    boxed = re.findall(r"\\boxed\{\(?([A-Da-d])\)?\}", text)
    if boxed:
        return boxed[-1].upper()

    # "Answer: A" / "final answer is (B)"
    answer_line = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*\(?([A-Da-d])\)?",
        text,
        flags=re.IGNORECASE,
    )
    if answer_line:
        return answer_line[-1].upper()

    # Bare (A) / A at end — take the last letter match
    letters = re.findall(r"\b([A-Da-d])\b", text)
    if letters:
        return letters[-1].upper()

    return ""


def _gold_letter(answer) -> str:
    """Normalize the gold answer to a letter.

    recursal/longbench-v2 stores ``answer`` as an integer index into
    ``choices`` (0->A, 1->B, ...). Also tolerate an already-lettered value.
    """
    if isinstance(answer, bool):
        return ""
    if isinstance(answer, int):
        return "ABCD"[answer] if 0 <= answer < 4 else ""
    s = str(answer).strip().upper()
    if s.isdigit():
        i = int(s)
        return "ABCD"[i] if 0 <= i < 4 else ""
    m = re.search(r"[A-D]", s)
    return m.group(0) if m else ""


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    gold = _gold_letter(doc.get("answer", ""))
    candidate = extract_choice(results[0] if results else "")
    return {"exact_match": int(bool(candidate) and candidate == gold)}
