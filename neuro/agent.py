import ast
import logging
from os.path import expanduser
import time
from typing import List


import joblib
import numpy as np
from tqdm import tqdm
import time
import functools

def retry(max_attempts=3, delay=1, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    logging.info(f"Attempt {attempts} failed with error: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator


def _extract_python_list_from_str(questions_list_str: str) -> List[str]:
    """Extract a Python list from a string representation."""
    start = questions_list_str.find('[')
    end = questions_list_str.rfind(']') + 1
    logging.info(f"Extracting Python list from string: {questions_list_str[start:end]}")
    # try:
    return ast.literal_eval(questions_list_str[start:end])
    # except (ValueError, SyntaxError) as e:
        # logging.error(f"Error extracting Python list: {e}")
        # logging.error(f"Let's try reformatting quotes in the input")
        

def brainstorm_init_questions(lm, args) -> List[str]:
    PROMPT = f"""
You are a scientific agent tasked with generating useful questions for linearly predicting fMRI responses to natural language stimuli.
{f'Specifically, you are predicting fMRI responses to the {args.predict_subset} cortex.' if not args.predict_subset == 'all' else ''}

Brainstorm some questions that could be useful.
Return a python list of strings and nothing else.
Each question must start with "Does the input" and end with "?".
Example: ["Does the input mention a location?", "Does the input mention time?", "Does the input contain a proper noun?"]
""".strip()
    questions_list_str = lm(PROMPT, max_completion_tokens=None, temperature=0) #, max_completion_tokens=1000, temperature=0)
    questions_list = _extract_python_list_from_str(questions_list_str)
    questions_list = remove_invalid_questions(questions_list)
    return questions_list

def _format_str_list_as_bullet_point_str(questions_list: List[str]) -> str:
    """Format a list of strings as bullet points."""
    return '\n'.join(f"- {question}" for question in questions_list)

def remove_invalid_questions(questions_list: List[str]) -> List[str]:
    """Ensure all questions start with 'Does the input' and end with '?'."""
    questions_list_revised = [
        q for q in questions_list
        if q.startswith('Does the input') and q.endswith('?')
    ]
    if len(questions_list_revised) < len(questions_list):
        logging.info("Some questions were removed because they did not start with 'Does the input' and end with '?'")
        for q in questions_list:
            if not (q.startswith('Does the input') and q.endswith('?')):
                logging.info(f"\tRemoved question: {q}")
    return questions_list_revised

@retry(max_attempts=5, delay=2)
def update_questions(lm, args, questions_list: List[str], r) -> List[str]:
    questions_arr = np.array(questions_list)
    qs_sort_idx = np.argsort(np.array(r['feature_importances_var_explained']))[::-1]
    questions_arr = questions_arr[qs_sort_idx]
    feature_importances = np.array(r['feature_importances_var_explained'])[qs_sort_idx]
    
    # extract topk tuples of questions that have the highest feature correlations from feature correlation matrix
    feature_correlation_matrix = r['feature_correlations']
    # set diag & triangle to -1 to avoid self pairs / duplicate pairs
    feature_correlation_matrix[np.triu(feature_correlation_matrix, k=0).astype(bool)] = -1
    topk_indices = np.unravel_index(np.argsort(feature_correlation_matrix, axis=None)[::-1][:args.topk_agent_correlated_questions], feature_correlation_matrix.shape)
    topk_questions_with_imp = []
    for i, j in zip(*topk_indices):
        if i != j:
            topk_questions_with_imp.append((questions_list[i], questions_list[j], feature_correlation_matrix[i, j]))

    top_error_ngrams = r['error_ngrams_df']['ngram'].values[:args.topk_agent_errors]



    PROMPT = f"""
# Main instructions
You are a scientific agent tasked with generating useful questions for linearly predicting fMRI responses to natural language stimuli.
{f'Specifically, you are predicting fMRI responses to the {args.predict_subset} cortex.' if not args.predict_subset == 'all' else ''}

Here is the original list of questions that have been previously tested, along with their feature importance (higher is more important):
--------------------
Question, Importance
--------------------
{'\n'.join(f"{question}, {importance:.3f}" for question, importance in zip(questions_arr, feature_importances))}
--------------------

# Supporting information
Here are the {args.topk_agent_correlated_questions} most correlated questions among the original list:
-----------------------------------
Question 1, Question 2, Correlation
-----------------------------------
{'\n'.join(f"{q1}, {q2}, {corr:.2f}" for (q1, q2, corr) in topk_questions_with_imp)}
-----------------------------------

Here are the {args.topk_agent_errors} text examples that are most poorly predicted by the original list of questions. Patterns that exist across many of these examples may suggest new questions to ask:
{_format_str_list_as_bullet_point_str(top_error_ngrams)}

# Final instructions
Extend and revise the original list of questions with several more questions that could be useful.
Merge questions that seem too similar.
Add new questions that capture potentially missing aspects.
Do not needlessly reword existing questions.
Output at least as many questions as there are in the input list, likely exactly repeating at least some of the questions.
Return a python list of strings and nothing else.
Each question must start with "Does the input" and end with "?". It is very important that every question starts with "Does the input".
Example output: ["Does the input mention a location?", "Does the input mention time?", "Does the input contain a proper noun?"]
""".strip()
    questions_list_str = lm(PROMPT, temperature=0, max_completion_tokens=None)
    questions_list = _extract_python_list_from_str(questions_list_str)
    logging.info(f"Updated questions list: {questions_list}")
    # assert all(q.startswith('Does the input') and q.endswith('?') for q in questions_list), \
        # "All questions must start with 'Does the input' and end with '?'"
    questions_list = remove_invalid_questions(questions_list)
    assert len(questions_list) >= len(questions_arr), \
        f"Updated questions list must have at least as many questions as the original list. Original length: {len(questions_arr)}, updated length: {len(questions_list)}"
    return questions_list
