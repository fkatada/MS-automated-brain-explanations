
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

import neuro.flatmaps_helper
import neuro.features.feature_spaces
import neuro.features.stim_utils


def nancorr(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    return np.corrcoef(x[mask], y[mask])[0, 1]


def evaluate_pc_model_on_each_voxel(
        args, stim, resp,
        model_params_to_save, pca, scaler,
        predict_subset='all',
):
    if args.encoding_model == 'ridge':
        weights_pc = model_params_to_save['weights_pc']
        preds_pc = stim @ weights_pc
        model_params_to_save['weights'] = weights_pc * \
            scaler.scale_ @ pca.components_
        model_params_to_save['bias'] = scaler.mean_ @ pca.components_ + pca.mean_
        # note: prediction = stim @ weights + bias
        # bias is optional, is really just zero
    elif args.encoding_model in ['tabpfn']:
        preds_pc = model_params_to_save['preds_pc']
        preds_pc = np.array(preds_pc).T
    elif args.encoding_model == 'mlp':
        preds_pc = model_params_to_save['preds_pc']

    preds_voxels = pca.inverse_transform(
        scaler.inverse_transform(preds_pc))  # (n_trs x n_voxels)
    if not predict_subset == 'all':
        idxs_mask = neuro.flatmaps_helper.load_custom_rois(
            args.subject.replace('UT', ''), '_lobes')[predict_subset]
        vox_idxs = np.where(idxs_mask)[0]
    else:
        vox_idxs = range(preds_voxels.shape[1])

    corrs = []
    for i in vox_idxs:
        corrs.append(nancorr(preds_voxels[:, i], resp[:, i]))
    corrs = np.array(corrs)
    corrs[np.isnan(corrs)] = 0
    return corrs

def explained_variance_per_feature(X, y, w, intercept=0.0, method="lofo"):
    """
    Compute per‑feature explained variance for an already‑fit Ridge (or any linear) model.
    
    Parameters
    ----------
    X : (n_samples, n_features) array_like
        Design matrix used to train the model.
    y : (n_samples,) array_like
        Target vector.
    w : (n_features,) array_like
        Learned weight vector.
    intercept : float, default 0.0
        Intercept term of the model.
    method : {"lofo", "variance", "covariance"}
        - "lofo"      : ΔR² when feature j is left out (unique importance).
        - "variance"  : Var(X_j w_j) / Var(y)  (assumes X columns are centred & mutually
                         uncorrelated).
        - "covariance": w_j * Cov(X_j, y) / Var(y) (can be negative).
    
    Returns
    -------
    contrib : (n_features,) ndarray
        Explained‑variance contribution for each feature.
    """
    X = X
    y = y.ravel()
    w = w.ravel()
    
    var_y = np.var(y, ddof=0)
    if var_y == 0:
        raise ValueError("y has zero variance.")
    
    # Common helper: full‑model predictions & R²
    y_hat_full = intercept + X @ w
    r2_full = 1 - np.var(y - y_hat_full, ddof=0) / var_y
    
    if method == "lofo":
        contrib = np.empty_like(w, dtype=float)
        for j in range(len(w)):
            y_hat_minus_j = y_hat_full - X[:, j] * w[j]
            r2_minus_j = 1 - np.var(y - y_hat_minus_j, ddof=0) / var_y
            contrib[j] = r2_full - r2_minus_j
        return contrib
    
    if method == "variance":
        # Ensure X columns are centred; otherwise centre them here.
        var_hat_j = np.var(X * w, axis=0, ddof=0)      # element‑wise product
        return var_hat_j / var_y
    
    if method == "covariance":
        y_centered = y - y.mean()
        cov_j = (X * (y_centered[:, None])).mean(axis=0)   # Cov(X_j, y)
        return w * cov_j / var_y
    
    raise ValueError("method must be 'lofo', 'variance', or 'covariance'")


def explained_var_over_targets_and_delays(args, stim_train_delayed, resp, model_params_to_save):
    """
    stim_train_delayed: (n_samples, n_features * args.ndelays)
    resp: (n_samples, n_targets)
    W: (n_features * args.ndelays, n_targets)

    Returns:
        mean_var_explained: (n_features,)
    """
    logging.info("Computing explained variance per feature over targets and delays...")
    # check stim shape
    n_weights = stim_train_delayed.shape[1]
    n_questions = len(args.qa_questions_version)
    assert n_weights == n_questions * args.ndelays, \
        f"stim.shape[1] ({stim_train_delayed.shape[1]}) != n_questions * args.ndelays ({n_questions * args.ndelays})"
    
    # select voxels to evaluate
    if not args.predict_subset == 'all':
        idxs_mask = neuro.flatmaps_helper.load_custom_rois(
            args.subject.replace('UT', ''), '_lobes')[args.predict_subset]
        vox_idxs = np.where(idxs_mask)[0]
    else:
        vox_idxs = range(resp.shape[1])
    n_targets = len(vox_idxs)

    # setup pred for each target
    var_explained = np.zeros((n_targets, n_weights))
    for i, i_single_vox in enumerate(tqdm(vox_idxs)):
        y_single_vox = resp[:, i_single_vox]
        
        # Covariance of each feature's contribution with total prediction
        w_single_vox = model_params_to_save['weights'][:, i_single_vox]
        var_explained[i] = explained_variance_per_feature(
            stim_train_delayed, y_single_vox, w_single_vox, method='covariance',
        )
        
    # average over delays
    var_explained = var_explained.reshape(n_targets, args.ndelays, n_questions)
    mean_var_explained = var_explained.mean(axis=0).mean(axis=0)

    return mean_var_explained

def get_ngrams_top_errors_df(story_names, stim_delayed, resp, model_params_to_save, num_top_errors = 300):
    errors_dict = defaultdict(list)
    wordseqs = neuro.features.stim_utils.load_story_wordseqs_huge(story_names)
    ngrams_list_total = []
    for story_name in tqdm(story_names):
        # ngram for 3 trs preceding the current TR
        chunks = wordseqs[story_name].chunks()
        ngrams_list = neuro.features.feature_spaces._get_ngrams_list_from_chunks(
            chunks, num_trs=3)
        ngrams_list = np.array(ngrams_list[10:-5])
        ngrams_list_total.extend(ngrams_list)

    preds = stim_delayed @ model_params_to_save['weights']

    # calculate correlation at each timepoint
    corrs_time = np.array([np.corrcoef(resp[i, :], preds[i, :])[0, 1]
                        for i in range(resp.shape[0])])
    corrs_time[:10] = 100  # don't pick first 10 TRs
    corrs_time[-10:] = 100  # don't pick last 10 TRs
    
    # get worst idxs
    corrs_worst_idxs = np.argsort(corrs_time)[:num_top_errors]        
    for i in range(num_top_errors):
        errors_dict['corrs'].append(corrs_time[corrs_worst_idxs[i]])
        errors_dict['ngram'].append(ngrams_list_total[corrs_worst_idxs[i]])
        errors_dict['tr'].append(corrs_worst_idxs[i])

    return pd.DataFrame(errors_dict)

def add_summary_stats(r, verbose=True):
    for key in ['corrs_test', 'corrs_tune', 'corrs_tune_pc', 'corrs_test_pc', 'corrs_brain_drive']:
        if key in r:
            r[key + '_mean'] = np.nanmean(r[key])
            r[key + '_median'] = np.nanmedian(r[key])
            r[key + '_frac>0'] = np.nanmean(r[key] > 0)
            r[key + '_mean_top1_percentile'] = np.nanmean(
                np.sort(r[key])[-len(r[key]) // 100:])
            r[key + '_mean_top5_percentile'] = np.nanmean(
                np.sort(r[key])[-len(r[key]) // 20:])

            # add r2 stats
            r[key.replace('corrs', 'r2') +
              '_mean'] = np.nanmean(r[key] * np.abs(r[key]))
            r[key.replace('corrs', 'r2') +
              '_median'] = np.nanmedian(r[key] * np.abs(r[key]))

            if key in ['corrs_test', 'corrs_brain_drive'] and verbose:
                logging.info(f"mean {key}: {r[key + '_mean']:.4f}")
                logging.info(f"median {key}: {r[key + '_median']:.4f}")
                logging.info(f"frac>0 {key}: {r[key + '_frac>0']:.4f}")
                logging.info(
                    f"mean top1 percentile {key}: {r[key + '_mean_top1_percentile']:.4f}")
                logging.info(
                    f"mean top5 percentile {key}: {r[key + '_mean_top5_percentile']:.4f}")

    return r
