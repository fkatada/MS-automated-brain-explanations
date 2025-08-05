from ridge_utils.DataSequence import DataSequence
import pandas as pd
import matplotlib.pyplot as plt
from os.path import dirname
import os
from tqdm import tqdm
from neuro.features import qa_questions, feature_spaces
from neuro.data import story_names, response_utils
from neuro.features.stim_utils import load_story_wordseqs, load_story_wordseqs_huge
import neuro.config
import seaborn as sns
import numpy as np
import joblib
from collections import defaultdict
from os.path import join
import warnings


def get_fmri_and_labs(story_name='onapproachtopluto', train_or_test='test', subject='uts03'):
    '''
    Returns
    -------
    df : pd.DataFrame
        The fMRI features, with columns corresponding to the principal components
        of the fMRI data.
    labs : pd.DataFrame
        Binary labeled annotations for each of the texts
    texts: 
        The texts corresponding to the rows of df
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        df = joblib.load(join(neuro.config.DECODING_DIR, f'{subject}/{train_or_test}/{story_name}.pkl'))
        dfs = []
        for offset in [1, 2, 3, 4]:
            df_offset = df.shift(-offset)
            df_offset.columns = [col + f'_{offset}' for col in df.columns]
            dfs.append(df_offset)
        df = pd.concat(dfs, axis=1)  # .dropna()  # would want to dropna here

        # load labels
        labs = joblib.load(join(neuro.config.DECODING_DIR, f'labels/{train_or_test}/{story_name}_labels.pkl'))

        # drop rows with nans
        idxs_na = df.isna().sum(axis=1).values > 0
        df = df[~idxs_na]
        labs = labs[~idxs_na]
        texts = pd.Series(df.index)
        return df, labs, texts


def concatenate_running_texts(texts, frac=1/2):
    '''When decoding, you might want to concatenate 
    the text of the current and surrounding texts
    to deal with the temporal imprecision of the fMRI signal.
    '''
    texts_before = (
        texts.shift(1)
        .str.split().apply(  # only keep second half of words
            lambda l: ' '.join(l[int(-len(l) * frac):]) if l else '')
    )

    texts_after = (
        texts.shift(-1)
        .str.split().apply(  # only keep first half of words
            lambda l: ' '.join(l[:int(len(l) * frac)]) if l else '')
    )

    return texts_before + ' ' + texts + ' ' + texts_after

def get_shared_data_for_subjects(subjects, train_or_test='train', concatenate_running_texts_frac=0.5):
    story_names = set(os.listdir(join(neuro.config.DECODING_DIR, f'{subjects[0]}/{train_or_test}')))
    for subject in subjects[1:]:
        story_names &= set(os.listdir(join(neuro.config.DECODING_DIR, f'{subject}/{train_or_test}')))

    datas = []
    for subject in subjects:
        data = defaultdict(list)
        for story_name in story_names:
        
            df, labs, texts = get_fmri_and_labs(
                story_name.replace('.pkl', ''), train_or_test, subject)
            data['df'].append(df)
            texts = concatenate_running_texts(texts, frac=concatenate_running_texts_frac)
            data['texts'].append(texts)

        for k in data:
            data[k] = pd.concat(data[k], axis=0)

        datas.append(data)

    # assert texts are the same across subjects
    texts = datas[0]['texts']
    for data in datas[1:]:
        assert (data['texts'] == texts).all(), "Texts do not match across subjects"

    # concatenate columns of dfs across subjects
    df = pd.concat([data['df'] for data in datas], axis=1)
    df.columns = [f"{subject}_{col}" for subject in subjects for col in data['df'].columns]

    return df, texts