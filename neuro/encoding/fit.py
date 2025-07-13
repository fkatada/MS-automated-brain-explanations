import argparse
import logging
import os
import os.path
import random
import time
from collections import defaultdict
from copy import deepcopy
from os.path import join

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
from neuro.encoding.ridge import bootstrap_ridge
from neuro.features import feat_select, feature_utils
from neuro.features.questions.gpt4 import QS_HYPOTHESES_COMPUTED


def fit_regression(args, r, features_train_delayed, resp_train, features_test_delayed, resp_test):
    if args.pc_components > 0:
        # if args.min_alpha > 0:
        # alphas = np.logspace(np.log10(args.min_alpha), 4, 12)
        # else:
        alphas = np.logspace(1, 4, 12)
        weights_key = 'weights_pc'
        corrs_key_test = 'corrs_test_pc'
        corrs_key_tune = 'corrs_tune_pc'
    else:
        # if args.min_alpha > 0:
        # alphas = np.logspace(np.log10(args.min_alpha), 4, 12)
        # else:
        alphas = np.logspace(1, 4, 12)
        weights_key = 'weights'
        corrs_key_test = 'corrs_test'
        corrs_key_tune = 'corrs_tune'

    if args.encoding_model == 'ridge':
        if args.use_test_setup == 3:
            example_params = {
                'features_train_delayed': features_train_delayed,
                'resp_train': resp_train,
                'features_test_delayed': features_test_delayed,
                'resp_test': resp_test,
                'alphas': alphas,
                'nboots': args.nboots,
                'chunklen': args.chunklen,
                'nchunks': args.nchunks,
                'singcutoff': args.singcutoff,
                'single_alpha': args.single_alpha,
            }
            joblib.dump(example_params, 'example_params_full.joblib')
        wt, corrs_test, alphas_best, corrs_tune, valinds = bootstrap_ridge(
            features_train_delayed, resp_train, features_test_delayed, resp_test,
            alphas, args.nboots, args.chunklen,
            args.nchunks, singcutoff=args.singcutoff, single_alpha=args.single_alpha)

        # Save regression results
        model_params_to_save = {
            weights_key: wt,
            'alphas_best': alphas_best,
            # 'valinds': valinds
        }

        # corrs_tune is (alphas, voxels, and bootstrap samples)
        # now reorder so it's (voxels, alphas, bootstrap samples)
        corrs_tune = np.swapaxes(corrs_tune, 0, 1)
        # mean over bootstrap samples
        corrs_tune = corrs_tune.mean(axis=-1)

        # replace each element of alphas_best with its index in alphas
        alphas_idx = np.array([np.where(alphas == a)[0][0]
                               for a in alphas_best])

        # apply best alpha to each voxel
        corrs_tune = corrs_tune[np.arange(corrs_tune.shape[0]), alphas_idx]

        # so we average over the bootstrap samples and take the max over the alphas
        r[corrs_key_tune] = corrs_tune
        r[corrs_key_test] = corrs_test
    elif args.encoding_model == 'randomforest':
        rf = RandomForestRegressor(
            n_estimators=100, n_jobs=10)  # , max_depth=5)
        corrs_test = []
        for i in range(resp_train.shape[1]):
            rf.fit(features_train_delayed, resp_train[:, i])
            preds = rf.predict(features_test_delayed)
            # corrs_test.append(np.corrcoef(resp_test[:, i], preds)[0, 1])
            corrs_test.append(nancorr(resp_test[:, i], preds[:, i]))
            print(i, 'rf corr', corrs_test[-1])
        corrs_test = np.array(corrs_test)
        corrs_test[np.isnan(corrs_test)] = 0
        r[corrs_key_test] = corrs_test
        model_params_to_save = {
            'weights': rf.feature_importances_,
        }
    elif args.encoding_model == 'tabpfn':
        from tabpfn import TabPFNRegressor
        rf = TabPFNRegressor(device='cuda')
        corrs_test = []
        preds_pc = []
        for i in tqdm(range(resp_train.shape[1])):
            rf.fit(features_train_delayed, resp_train[:, i])
            preds = rf.predict(features_test_delayed)
            corrs_test.append(nancorr(resp_test[:, i], preds))
            # print(i, 'tabpfn corr', corrs_test[-1])
            preds_pc.append(preds)
        corrs_test = np.array(corrs_test)
        corrs_test[np.isnan(corrs_test)] = 0
        r[corrs_key_test] = corrs_test
        model_params_to_save = {'preds_pc': preds_pc}
    elif args.encoding_model == 'mlp':
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(max_iter=1000)
        corrs_test = []
        mlp.fit(features_train_delayed, resp_train)
        preds = mlp.predict(features_test_delayed)
        for i in range(resp_train.shape[1]):
            corrs_test.append(nancorr(resp_test[:, i], preds[:, i]))
            # print(i, 'mlp corr', corrs_test[-1])
        corrs_test = np.array(corrs_test)
        corrs_test[np.isnan(corrs_test)] = 0
        r[corrs_key_test] = corrs_test
        model_params_to_save = {
            'preds_pc': preds,
        }

    return r, model_params_to_save