import os
from os.path import join

import dvu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# sys.path.append('../notebooks')
# flatmaps_per_question = __import__('06_flatmaps_per_question')

REMAP_CATEGORY_TO_QUESTIONS = {
    'Visuospatial':  (0, {
        13: 'Does the sentence describe a visual experience or scene?',
        18: 'Does the sentence describe a sensory experience?',
        15: 'Does the sentence involve spatial reasoning?',
        2: 'Does the sentence describe a physical action?',
        5: 'Does the sentence involve a description of physical environment or setting?',
        # questionable
        4: 'Does the sentence involve the mention of a specific object or item?',
        35: 'Does the sentence involve a planning or decision-making process?',
        36: 'Does the sentence involve a discussion about future plans or intentions?',
        -1: 'Does the sentence contain words with strong visual imagery?',
    }),
    'Communication': (1, {
        3: 'Does the sentence describe a personal or social interaction that leads to a change or revelation?',
        6: 'Does the sentence describe a relationship between people?',
        22: 'Does the text describe a mode of communication?',
        12: 'Does the sentence include dialogue?',
        32: 'Does the sentence include a direct speech quotation?',
        -2: 'Does the sentence involve a description of an interpersonal misunderstanding or dispute?',
    }),
    'Beliefs, values, emotions': (2, {
        23: 'Does the input include a comparison or metaphor?',
        10: "Does the sentence express the narrator's opinion or judgment about an event or character?",
        16: 'Does the sentence involve an expression of personal values or beliefs?',
        28: 'Does the sentence involve a discussion about personal or social values?',
        17: 'Does the sentence contain a negation?',
        9: 'Is the sentence abstract rather than concrete?',
        0: 'Does the sentence describe a personal reflection or thought?',
        33: 'Is the sentence reflective, involving self-analysis or introspection?',
        -3: 'Does the sentence involve the description of an emotional response?',
       -4: 'Does the sentence use irony or sarcasm?',
    }),
    'Numeric': (3, {
        8: 'Is time mentioned in the input?',
        20: 'Does the input contain a number?',
        30: 'Does the input contain a measurement?',
    }),
    'Tactile': (4, {
        31: 'Does the sentence describe a physical sensation?',
        34: 'Does the input describe a specific texture or sensation?',
        25: 'Does the sentence describe a specific sensation or feeling?',  
    }),
    'Other': (5, {
        1: 'Does the sentence contain a proper noun?',
        7: 'Does the sentence mention a specific location?',
        11: 'Is the input related to a specific industry or profession?',
        14: 'Does the input involve planning or organizing?',
        19: 'Does the sentence include technical or specialized terminology?',
        21: 'Does the sentence contain a cultural reference?',
        24: 'Does the sentence express a sense of belonging or connection to a place or community?',
        26: 'Does the text include a planning or decision-making process?',
        27: 'Does the sentence include a personal anecdote or story?',
        29: 'Does the text describe a journey?',
       
       -5: 'Is there a first-person pronoun in the input?',
       -6: 'Is the sentence part of a legal document or text?',
       -7: 'Does the input describe a scientific experiment or discovery?',
       -8: 'Does the input discuss a breakthrough in medical research?',
       -9: 'Does the input involve a coding or programming concept?',
       -10: 'Is an educational lesson or class described?'
    }),
    # 'Planning': (6, {
    #     35: 'Does the sentence involve a planning or decision-making process?',
    #     36: 'Does the sentence involve a discussion about future plans or intentions?',
    # }),
}
REMAP_QUESTIONS_TO_CATEGORY_IDXS = {}
for category, (idx, questions_dict) in REMAP_CATEGORY_TO_QUESTIONS.items():
    for q_idx, q in questions_dict.items():
        REMAP_QUESTIONS_TO_CATEGORY_IDXS[q] = idx

REMAP_QUESTIONS_TO_CATEGORY_NAMES = {}
for category, (idx, questions_dict) in REMAP_CATEGORY_TO_QUESTIONS.items():
    for q_idx, q in questions_dict.items():
        REMAP_QUESTIONS_TO_CATEGORY_NAMES[q] = category

def compute_pvals(flatmaps_qa_list, frac_voxels_to_keep, corrs_gt_arr, flatmaps_null, mask_corrs=None):
    '''
    Params
    ------
    flatmaps_qa_list: list of D np.arrays
        each array is a flatmap of the same shape
    frac_voxels_to_keep: float
        fraction of voxels to keep
    corrs_gt_arr: np.array of size D
        array of ground truth correlations 
    flatmaps_null: np.ndarray
        (n_flatmaps, n_voxels) array of null flatmaps for a particular subject
        e.g. eng1000 weights
    mask_corrs: np.ndarray
        if passed, use this as mask rather than mask extreme
    '''
    # print(eng1000_dir)
    # flatmaps_eng1000 = joblib.load(eng1000_dir)
    pvals = []
    baseline_distr = []

    for i in tqdm(range(len(flatmaps_qa_list))):
        if frac_voxels_to_keep < 1:

            # mask based on corrs
            if mask_corrs is not None:
                mask = (mask_corrs > np.percentile(
                    mask_corrs, 100 * (1 - frac_voxels_to_keep))).astype(bool)
            else:
                # mask based on extreme values
                mask = np.abs(flatmaps_qa_list[i]) >= np.percentile(
                    np.abs(flatmaps_qa_list[i]), 100 * (1 - frac_voxels_to_keep))

        else:
            mask = np.ones_like(flatmaps_qa_list[i]).astype(bool)
        # if frac_voxels_to_keep < 1:
        #     mask_extreme = np.abs(flatmaps_qa_list[i]) >= np.percentile(
        #         np.abs(flatmaps_qa_list[i]), 100 * (1 - frac_voxels_to_keep))
        # else:
        #     mask_extreme = np.ones(flatmaps_qa_list[i].shape).astype(bool)

        flatmaps_null_masked = flatmaps_null[:, mask]
        flatmaps_qa_masked = flatmaps_qa_list[i][mask]

        # calculate correlation between each row of flatmaps_qa_masked and flatmaps_eng1000_masked
        flatmaps_null_masked_norm = StandardScaler(
        ).fit_transform(flatmaps_null_masked.T).T
        flatmaps_qa_masked_norm = (
            flatmaps_qa_masked - flatmaps_qa_masked.mean()) / flatmaps_qa_masked.std()
        corrs_perm_eng100_arr = flatmaps_null_masked_norm @ flatmaps_qa_masked_norm / \
            flatmaps_qa_masked_norm.shape[0]
        pvals.append((corrs_perm_eng100_arr > corrs_gt_arr[i]).mean())
        baseline_distr.append(corrs_perm_eng100_arr)
    return pvals, baseline_distr


def _calc_corrs(flatmaps_qa, flatmaps_gt, titles_qa, titles_gt, preproc=None):
    if preproc is not None:
        if preproc == 'quantize':
            # bin into n bins with equal number of samples
            n_bins = 10

    corrs = pd.DataFrame(
        np.zeros((len(flatmaps_qa), len(flatmaps_gt))),
        index=titles_qa,
        columns=titles_gt,
        # [f'{bd_list[i][0]}_{bd_list[i][1]}'.replace('_qa', '') for i in range(len(bd_list))],
        # index=df_pairs['qa_weight'].astype(str),
    )
    for i, qa in enumerate(flatmaps_qa):
        for j, bd in enumerate(flatmaps_gt):
            corrs.iloc[i, j] = np.corrcoef(
                flatmaps_qa[i], flatmaps_gt[j])[0, 1]

    return corrs


def _heatmap(corrs, out_dir_save):
    os.makedirs(out_dir_save, exist_ok=True)
    # normalize each column
    # corrs = corrs / corrs.abs().max()
    # normalize each row to mean zero stddev 1
    # corrs = (corrs - corrs.mean()) / corrs.std()
    # plt.figure(figsize=(20, 10))
    vmax = np.max(np.abs(corrs.values))
    # sns.clustermap(corrs, annot=False, cmap='RdBu', vmin=-vmax, vmax=vmax, dendrogram_ratio=0.01)

    sns.heatmap(corrs, annot=False, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    # plt.imshow(corrs, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    # plt.xticks(range(len(bd_list)), [f'{bd[0]}' for bd in bd_list], rotation=90)
    # plt.yticks(range(len(qa_list)), qa_list)
    # plt.colorbar()
    dvu.outline_diagonal(corrs.values.shape, roffset=0.5, coffset=0.5)
    plt.savefig(join(out_dir_save, 'corrs_heatmap.pdf'),
                bbox_inches='tight')


def corr_bars(corrs: np.ndarray, questions, out_dir_save, xlab: str = '', color='C0', label=None):
    os.makedirs(out_dir_save, exist_ok=True)
    # print(out_dir_save)
    # mask = args0['corrs_test'] >= 0
    # wt_qas = [wt_qa[mask] for wt_qa in wt_qas]
    # wt_bds = [wt_bd[mask] for wt_bd in wt_bds]

    # barplot of diagonal
    def _get_func_with_perm(x1, x2, n=10, perc=95):
        corr = np.corrcoef(x1, x2)[0, 1]
        corrs_perm = [
            np.corrcoef(np.random.permutation(x1), x2)[0, 1]
            for i in range(n)
        ]
        # print(corrs_perm)
        return corr, np.percentile(corrs_perm, 50-perc/2), np.percentile(corrs_perm, 50+perc/2)

    # corrs_mean = []
    # corrs_err = []
    # for i in tqdm(range(len(corrs.columns))):
    #     corr, corr_interval_min, corr_interval_max = _get_func_with_perm(
    #         flatmaps_qa[i], flatmaps_gt[i])
    #     corrs_mean.append(corr)
    #     corrs_err.append((corr_interval_max - corr_interval_min)/2)
    # sns.barplot(y=corrs.columns, x=np.diag(corrs), color='gray')
    plt.grid(alpha=0.2)
    # corrs_diag = np.diag(corrs)
    # idx_sort = np.argsort(corrs_diag)[::-1]
    # idx_sort = np.arange(len(questions))
    # print('idx_sort', idx_sort, corrs_diag[idx_sort])
    plt.errorbar(
        x=corrs,
        y=np.arange(len(questions)),
        # xerr=corrs_err,
        fmt='o',
        color=color,
        label=label + f' (mean {corrs.mean():.3f})',
    )

    plt.yticks(np.arange(len(questions)), questions)
    plt.axvline(0, color='gray')
    plt.axvline(corrs.mean(), color=color, linestyle='--')
    # plt.title(f'{setting} mean {np.diag(corrs).mean():.3f}')
    # annotate line with mean value
    # plt.text(np.diag(corrs).mean(), 0.1,
    #  f'{np.diag(corrs).mean():.3f}', ha='left', color=color)
    plt.xlabel(xlab)
