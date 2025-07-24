import ast
import argparse
import logging
import os
import os.path
import random
from os.path import expanduser
import time
from collections import defaultdict
from copy import deepcopy
from os.path import join
from typing import List

import imodelsx.cache_save_utils
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

import neuro.data.story_names as story_names
from neuro.data import response_utils
from neuro.encoding.eval import (
    add_summary_stats,
    evaluate_pc_model_on_each_voxel,
    nancorr,
)
from neuro.encoding.fit import fit_regression
from neuro.features import feat_select, feature_utils
from neuro.features.questions.gpt4 import QS_HYPOTHESES_COMPUTED


def _extract_python_list_from_str(questions_list_str: str) -> List[str]:
    """Extract a Python list from a string representation."""
    start = questions_list_str.find('[')
    end = questions_list_str.rfind(']') + 1
    return ast.literal_eval(questions_list_str[start:end])

def brainstorm_init_questions(lm, args) -> List[str]:
    PROMPT = f"""
You are a scientific agent tasked with generating useful questions for linearly predicting fMRI responses to natural language stimuli.
{f'Specifically, you are predicting fMRI responses to the {args.predict_subset} cortex.' if not args.predict_subset == 'all' else ''}

Brainstorm some questions that could be useful.
Return a python list of strings and nothing else.
Each question should start with "Does the input" and end with "?".
Example: ['Does the input mention a location?', 'Does the input mention time?', 'Does the input contain a proper noun?']
""".strip()
    questions_list_str = lm(PROMPT, max_completion_tokens=None, temperature=0) #, max_completion_tokens=1000, temperature=0)
    return _extract_python_list_from_str(questions_list_str)

def _format_str_list_as_bullet_point_str(questions_list: List[str]) -> str:
    """Format a list of strings as bullet points."""
    return '\n'.join(f"- {question}" for question in questions_list)

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
Extend and revise the original list of questions with more questions that could be useful.
Merge questions that seem too similar.
Add new questions that capture potentially missing aspects.
Do not needlessly reword existing questions.
Output at least as many questions as there are in the input list, likely exactly repeating at least some of the questions.
Return a python list of strings and nothing else.
Each question should start with "Does the input" and end with "?".
Example output: ['Does the input mention a location?', 'Does the input mention time?', 'Does the input contain a proper noun?']
""".strip()
    questions_list_str = lm(PROMPT, temperature=0)
    questions_list = _extract_python_list_from_str(questions_list_str)
    assert all(q.startswith('Does the input') and q.endswith('?') for q in questions_list), \
        "All questions must start with 'Does the input' and end with '?'"
    return questions_list
