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
You are a scientific agent tasked with generating useful questions for predicting fMRI responses to natural language stimuli.
{f'Specifically, you are predicting fMRI responses to the {args.predict_subset} cortex.' if not args.predict_subset == 'all' else ''}.

Brainstorm some questions that could be useful.
Return a python list of strings and nothing else. Each question should start with "Does the input" and end with "?".
Example: ['Does the input mention a location?', 'Does the input mention time?', 'Does the input contain a proper noun?']
""".strip()
    questions_list_str = lm(PROMPT, max_completion_tokens=1000, temperature=0)
    return _extract_python_list_from_str(questions_list_str)

def format_str_list_as_bullet_point_str(questions_list: List[str]) -> str:
    """Format a list of strings as bullet points."""
    return '\n'.join(f"- {question}" for question in questions_list)

def update_questions(lm, args, questions_list) -> List[str]:
    PROMPT = f"""
You are a scientific agent tasked with generating useful questions for predicting fMRI responses to natural language stimuli.
{f'Specifically, you are predicting fMRI responses to the {args.predict_subset} cortex.' if not args.predict_subset == 'all' else ''}.

Here are the previous questions that have been tested, along with their importance (higher is more important):
{format_str_list_as_bullet_point_str(questions_list)}

Extend and revise this list of questions with more questions that could be useful.
Merge questions that seem too similar.
Add new questions that capture potentially missing aspects.
Do not needlessly reword existing questions.
Output at least as many questions as there are in the input list, likely exactly repeating at least some of the questions.
Return a python list of strings and nothing else. Each question should start with "Does the input" and end with "?".
Example: ['Does the input mention a location?', 'Does the input mention time?', 'Does the input contain a proper noun?']
""".strip()
    questions_list_str = lm(PROMPT, max_completion_tokens=1000, temperature=0)
    questions_list = _extract_python_list_from_str(questions_list_str)
    assert all(q.startswith('Does the input') and q.endswith('?') for q in questions_list), \
        "All questions must start with 'Does the input' and end with '?'"
    return questions_list
