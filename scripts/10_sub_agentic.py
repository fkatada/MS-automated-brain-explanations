import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

params_shared_dict = {
    'use_test_setup': [0],
    'use_extract_only': [0],
    'pc_components': [100],
    'ndelays': [8],
    'num_stories': [-1], 

    'feature_space': ['qa_agent'],
    'qa_embedding_model': ['meta-llama/Meta-Llama-3-8B-Instruct'],


    'subject': [f'UTS0{k}' for k in range(1, 4)],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/jul24_2025_agentic_test'],

    # 16 is good for 8B model with 45 GB
    # 64 is good for 8B model with 2x45 GB
    # 256 works but is slower for 8B model with 4x45 GB
    'qa_batch_size': [64], 
}

params_coupled_dict = {}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, 'experiments', '02_fit_encoding.py')
amlt_kwargs = {
    # change this to run a cpu job
    'amlt_file': join(repo_dir, 'scripts', 'launch.yaml'),
    # 'sku': 'G1-A100',
    'sku': 'G2-A100',
    # 'target___name': 'msrresrchvc',
    'target___name': 'msroctovc',

    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}

submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    unique_seeds='seed_stories',
    amlt_kwargs=amlt_kwargs,
    # n_cpus=8,
    # actually_run=False,
    # gpu_ids=[[2, 3]],
    repeat_failed_jobs=True,
    shuffle=True,
    # cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; .venv/bin/python',
)
