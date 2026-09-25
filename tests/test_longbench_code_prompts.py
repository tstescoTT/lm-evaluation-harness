"""Regressions for the prefixed Xnhyacinth LongBench-E code documents."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
# Load the actual renderer without importing unrelated model adapters.
SPEC = importlib.util.spec_from_file_location(
    "harness_utils", ROOT / "lm_eval/utils.py"
)
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)
INSTRUCTION = "Please complete the code given below. \n"


@pytest.fixture(params=["lcc_e", "repobench-p_e"])
def rendered(request):
    config = UTILS.load_yaml_config(
        ROOT / "lm_eval/tasks/longbench" / f"{request.param}.yaml", mode="simple"
    )
    doc = {"context": INSTRUCTION + "\nclass Example:\n", "question": "    value = 1\n"}
    return UTILS.apply_template(config["doc_to_text"], doc), request.param, doc


def test_instruction_from_dataset_is_not_duplicated(rendered):
    prompt, _, _ = rendered
    assert prompt.startswith(INSTRUCTION)
    assert not prompt.startswith(INSTRUCTION * 2)
    assert prompt.count(INSTRUCTION) == 1


def test_next_line_cue_keeps_its_terminal_newline_and_context(rendered):
    prompt, task, doc = rendered
    body = doc["context"] + (doc["question"] if task == "repobench-p_e" else "")
    assert prompt.endswith("Next line of code:\n")
    assert prompt == body + "Next line of code:\n"
