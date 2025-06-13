from typing import Dict, List, Any
import re
import json
import base64
import zlib
import multiprocessing
import numpy as np
from collections import defaultdict
import logging
import pickle

# Configure logger
logger = logging.getLogger(__name__)

# Helper function for multiprocessing
def _temp_run_helper(sample, generation, debug, result, metadata_list, timeout):
    try:
        from lm_eval.tasks.livecodebench.testing_util import run_test
    except ImportError:
        from .testing_util import run_test
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def extract_code_generation(model_output: str, model_type: str = 'chat'):
    """Extract code from model output based on model type."""
    outputlines = model_output.split('\n')

    if model_type == 'base':
        return model_output.strip()
    elif model_type == 'chat':
        indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
        
        # If we found code blocks, extract the code
        if len(indexlines) >= 2:
            return '\n'.join(outputlines[indexlines[0] + 1:indexlines[1]])
        
        # If no code blocks found, return empty string
        return ''
    else:
        raise ValueError(f'Invalid model type: {model_type}')


def codegen_check_correctness(sample, generation, timeout, debug=False):
    """Check correctness of code generation with a global timeout."""
    
    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        try:
            from lm_eval.tasks.livecodebench.testing_util import run_test
        except ImportError:
            from .testing_util import run_test
        res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    global_timeout = (timeout + 1) * len(json.loads(sample['input_output'])['inputs'])
    if debug:
        logger.info(f'global timeout = {global_timeout}')
    p.join(timeout=global_timeout)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample['input_output'])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs['inputs']))]]
        if debug:
            logger.info('global timeout occured: alarm went off')
    return result[0], metadata_list[0]


def evaluate_generations_by_problem(problem_generations: list, sample: list, debug: bool, timeout: int):
    """Evaluate each problem.

    Args:
        problem_generations:
        sample:
        debug:
        timeout
    """
    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = codegen_check_correctness(sample, o, timeout=timeout, debug=debug)
            if debug:
                logger.info(f'\nSuccessful compilation of task {o_idx}!')
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    logger.info(f'Results were not True for all test cases {curr_res=}\n')
        except Exception as e:
            if debug:
                logger.info(f'Compilation failed, test framework exception = {repr(e)}{e}\n')
            # break
            curr_metadata = {}
        finally:
            assert isinstance(curr_res, list)
            assert isinstance(curr_metadata, dict)
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            logger.info(f'Sample\n{r}\nResult\n{res[i]}')
            logger.info('*' * 30 + '\n\n')
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,  # This parameter will be unused
    timeout=6,
):
    """We take the list of code generations and try to compile them and the run
    their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS
            dataset)
        level: difficulty level used in the generation, can be "all",
            "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is
            a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test
            case [True] = passed test case
    """
    results = {}
    metadata = {}

    for index in range(len(generations_list)):
        problem_generations = generations_list[index]
        sample = samples_list[index]

        result, meta = evaluate_generations_by_problem(problem_generations, sample, debug, timeout)
        results[index] = result
        metadata[index] = meta

    assert len(results) == len(
        generations_list), f'results = {len(results)} inputs = {len(generations_list)} {results=}'

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
    try:
        from lm_eval.tasks.livecodebench.pass_k_utils import compute_metrics_from_results
    except ImportError:
        try:
            # Fallback for relative import
            from .pass_k_utils import compute_metrics_from_results
        except ImportError:
            # Last fallback: try to import from same directory
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from pass_k_utils import compute_metrics_from_results

    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(zip(samples_list, generations_list)):
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        assert len(final_metadata[i]) == len(generations_list[0]), f'{len(final_metadata[i])=}'

    return [metrics, results, final_metadata]


def transform_data_item(item):
    """Transform a single data item to match evalscope format."""
    # Define the format prompt constants
    FORMATTING_MESSAGE_WITH_STARTER_CODE = 'You will use the following starter code to write the solution to the problem and enclose your code within delimiters.'
    FORMATTING_WITHOUT_STARTER_CODE = 'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.'
    
    # starter_code
    if item.get('starter_code'):
        format_prompt = f'### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'
        format_prompt += f"```python\n{item['starter_code']}\n```\n\n"
    else:
        format_prompt = f'### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n'
        format_prompt += '```python\n# YOUR CODE HERE\n```\n\n'

    item['format_prompt'] = format_prompt

    # load test cases
    public_test_cases = item.get('public_test_cases', '[]')
    try:
        public_test_cases = json.loads(public_test_cases)
    except:
        public_test_cases = []

    private_test_cases = item.get('private_test_cases', '[]')
    try:
        private_test_cases = json.loads(private_test_cases)
    except Exception:
        try:
            # Handle compressed/pickled private test cases
            private_test_cases = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(private_test_cases.encode('utf-8')))))
        except Exception:
            private_test_cases = []

    # load metadata
    metadata = item.get('metadata', '{}')
    try:
        metadata = json.loads(metadata)
    except:
        metadata = {}
    
    evaluation_sample = json.dumps({
        'inputs': [t['input'] for t in public_test_cases + private_test_cases],
        'outputs': [t['output'] for t in public_test_cases + private_test_cases],
        'fn_name': metadata.get('func_name', None),
    })
    item['evaluation_sample'] = evaluation_sample

    return item


# --- lm-eval task-specific functions ---

def filter_by_difficulty(docs, difficulty_filter):
    """
    Filter documents by difficulty levels.
    
    Args:
        docs: List of documents/problems
        difficulty_filter: List of difficulty levels to include (e.g., ["Easy", "Medium", "Hard"])
                          If None or empty, no filtering is applied
    
    Returns:
        Filtered list of documents
    """
    if not difficulty_filter:
        logger.info("No difficulty filter specified - including all problems")
        return docs
    
    # Normalize difficulty filter values (handle case variations)
    normalized_filter = [d.lower().strip() for d in difficulty_filter]
    
    original_count = len(docs)
    filtered_docs = []
    
    for doc in docs:
        doc_difficulty = doc.get('difficulty', '').lower().strip()
        
        # Handle various difficulty naming conventions
        if doc_difficulty in normalized_filter:
            filtered_docs.append(doc)
        # Handle AtCoder contest type mapping to difficulty
        elif 'contest_type' in doc:
            contest_type = doc.get('contest_type', '').lower().strip()
            mapped_difficulty = None
            
            # Map AtCoder contest types to standard difficulties
            if 'abc' in contest_type:
                mapped_difficulty = 'easy'
            elif 'arc' in contest_type:
                mapped_difficulty = 'medium'  
            elif 'agc' in contest_type:
                mapped_difficulty = 'hard'
                
            if mapped_difficulty and mapped_difficulty in normalized_filter:
                filtered_docs.append(doc)
    
    filtered_count = len(filtered_docs)
    logger.info(f"Difficulty filtering: {original_count} problems → {filtered_count} problems")
    logger.info(f"Included difficulties: {difficulty_filter}")
    logger.info(f"Problems per difficulty:")
    
    # Count problems by difficulty for logging
    difficulty_counts = {}
    for doc in filtered_docs:
        doc_difficulty = doc.get('difficulty', 'Unknown')
        difficulty_counts[doc_difficulty] = difficulty_counts.get(doc_difficulty, 0) + 1
    
    for diff, count in difficulty_counts.items():
        logger.info(f"  - {diff}: {count} problems")
    
    return filtered_docs

def doc_to_target(doc: dict) -> dict:
    """
    Returns the document with properly formatted input_output field.
    Uses the same transformation logic as evalscope.
    """
    logger.debug(f"Processing document: {doc.get('question_id', 'unknown')}")
    
    # Extract and process public and private test cases
    public_test_cases = doc.get('public_test_cases', [])
    private_test_cases = doc.get('private_test_cases', [])
    
    # Make a copy to avoid modifying the original
    processed_doc = doc.copy()
    
    # Transform the document using evalscope logic
    transformed_doc = transform_data_item(processed_doc)
    
    # The evaluation_sample field becomes the input_output field
    transformed_doc['input_output'] = transformed_doc['evaluation_sample']
    
    return transformed_doc


def postprocess_generation(model_output: str) -> str:
    """Extracts the generated code from the model's output."""
    return extract_code_generation(model_output)


def process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    """
    Processes the results for a single document and calculates accuracy.

    :param doc: The document dictionary.
    :param results: A list of model generations (typically one for pass@1).
    :return: A dictionary with the accuracy metric.
    """
    if not results:
        return {"acc": 0.0}

    # We typically evaluate the first generation for pass@1
    generated_code = postprocess_generation(results[0])

    # The `codegen_metrics` function expects lists of samples and generations
    samples_list = [doc]
    generations_list = [[generated_code]]

    timeout = 6 
    debug = False

    try:
        metrics, eval_results, final_metadata = codegen_metrics(
            samples_list=samples_list,
            generations_list=generations_list,
            k_list=[1],
            num_process_evaluate=1,
            timeout=timeout,
            debug=debug,
        )
        # Extract pass@1 and convert from percentage to decimal
        pass_at_1 = metrics.get("pass@1", 0.0)
        accuracy = pass_at_1 / 100.0
        
        logger.debug(f"Pass@1: {pass_at_1}, Accuracy: {accuracy}")
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error during livecodebench metric calculation for doc_id {doc.get('id', 'unknown')}: {e}")
        logger.error(f"Full traceback: {error_traceback}")
        accuracy = 0.0

    return {"acc": accuracy}

# --- End of utility functions ---

class LiveCodeBenchEasyEfficient:
    """
    Efficient custom task class that filters LiveCodeBench to only Easy difficulty problems.
    This class filters at the dataset level before creating documents, making it much more efficient.
    """
    
    def __init__(self, **kwargs):
        try:
            from lm_eval.api.task import ConfigurableTask
        except ImportError:
            from lm_eval.base import ConfigurableTask
        
        self.config = kwargs
        self._base_task = ConfigurableTask(**kwargs)
        
        # Copy all attributes from the base task
        for attr in dir(self._base_task):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(self._base_task, attr))
    
    def __getattr__(self, name):
        """Delegate any missing attributes to the base task."""
        return getattr(self._base_task, name)
    
    def download(self, *args, **kwargs):
        """Override download to filter dataset at the HuggingFace dataset level."""
        # Call the base task's download method to get the raw dataset
        dataset = self._base_task.download(*args, **kwargs)
        
        # Filter the dataset efficiently at the HuggingFace dataset level
        logger.info("Filtering LiveCodeBench dataset for Easy difficulty problems only...")
        filtered_dataset = filter_dataset_by_difficulty(dataset, ["easy"])
        
        return filtered_dataset
    
    def eval_docs(self):
        """Override to use the already-filtered dataset."""
        # Since we filtered at the dataset level in download(), 
        # eval_docs() will naturally only return easy problems
        return self._base_task.eval_docs()

# Keep the old class for backward compatibility
class LiveCodeBenchEasy:
    """
    DEPRECATED: Use LiveCodeBenchEasyEfficient instead.
    This class filters documents after loading all of them, which is inefficient.
    """
    
    def __init__(self, **kwargs):
        try:
            from lm_eval.api.task import ConfigurableTask
        except ImportError:
            from lm_eval.base import ConfigurableTask
        
        self.config = kwargs
        self._base_task = ConfigurableTask(**kwargs)
        
        # Copy all attributes from the base task
        for attr in dir(self._base_task):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(self._base_task, attr))
    
    def __getattr__(self, name):
        """Delegate any missing attributes to the base task."""
        return getattr(self._base_task, name)
    
    def eval_docs(self):
        """Override to filter documents by difficulty."""
        # Get all documents from the base task
        all_docs = list(self._base_task.eval_docs())
        
        logger.info(f"Total documents before filtering: {len(all_docs)}")
        
        # Filter to only Easy difficulty
        easy_docs = []
        difficulty_counts = {}
        
        for doc in all_docs:
            doc_difficulty = doc.get('difficulty', '').lower().strip()
            difficulty_counts[doc_difficulty] = difficulty_counts.get(doc_difficulty, 0) + 1
            
            if doc_difficulty == 'easy':
                easy_docs.append(doc)
        
        logger.info(f"Documents after filtering to Easy only: {len(easy_docs)}")
        logger.info("Difficulty distribution in original dataset:")
        for diff, count in sorted(difficulty_counts.items()):
            if diff:  # Skip empty difficulties
                logger.info(f"  - {diff}: {count} problems")
        
        return easy_docs

def filter_dataset_by_difficulty(dataset, difficulty_filter):
    """
    Filter HuggingFace dataset by difficulty at dataset level (before converting to documents).
    This is much more efficient than filtering after loading all documents.
    
    Args:
        dataset: HuggingFace dataset
        difficulty_filter: List of difficulty levels to include (e.g., ["easy"])
                          If None or empty, no filtering is applied
    
    Returns:
        Filtered HuggingFace dataset
    """
    if not difficulty_filter:
        logger.info("No difficulty filter specified - including all problems")
        return dataset
    
    # Normalize difficulty filter values (handle case variations)
    normalized_filter = [d.lower().strip() for d in difficulty_filter]
    
    def filter_by_difficulty(example):
        """Filter function for HuggingFace dataset.filter()"""
        doc_difficulty = example.get('difficulty', '').lower().strip()
        
        # Direct difficulty match
        if doc_difficulty in normalized_filter:
            return True
            
        # Handle AtCoder contest type mapping to difficulty
        if 'contest_type' in example:
            contest_type = example.get('contest_type', '').lower().strip()
            mapped_difficulty = None
            
            # Map AtCoder contest types to standard difficulties
            if 'abc' in contest_type:
                mapped_difficulty = 'easy'
            elif 'arc' in contest_type:
                mapped_difficulty = 'medium'  
            elif 'agc' in contest_type:
                mapped_difficulty = 'hard'
                
            if mapped_difficulty and mapped_difficulty in normalized_filter:
                return True
        
        return False
    
    original_count = len(dataset)
    
    # Filter the dataset efficiently at the HuggingFace dataset level
    filtered_dataset = dataset.filter(filter_by_difficulty)
    
    filtered_count = len(filtered_dataset)
    logger.info(f"Difficulty filtering: {original_count} problems → {filtered_count} problems")
    logger.info(f"Included difficulties: {difficulty_filter}")
    
    # Count problems by difficulty for logging (sample a few to avoid loading all)
    if filtered_count > 0:
        sample_size = min(100, filtered_count)  # Sample to avoid loading entire dataset
        sample_data = filtered_dataset.select(range(sample_size))
        difficulty_counts = {}
        for example in sample_data:
            doc_difficulty = example.get('difficulty', 'Unknown')
            difficulty_counts[doc_difficulty] = difficulty_counts.get(doc_difficulty, 0) + 1
        
        logger.info(f"Sample difficulty distribution (first {sample_size} problems):")
        for diff, count in difficulty_counts.items():
            logger.info(f"  - {diff}: {count} problems")
    
    return filtered_dataset

def preprocess_easy_only(dataset):
    """
    Preprocessing function to filter LiveCodeBench dataset for easy questions only.
    This is called during dataset loading, making it much more efficient than post-loading filtering.
    
    Args:
        dataset: HuggingFace dataset
        
    Returns:
        Filtered dataset containing only easy questions
    """
    return filter_dataset_by_difficulty(dataset, ["easy"])