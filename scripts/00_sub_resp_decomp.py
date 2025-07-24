import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

params_shared_dict = {
    # 'subject': [f'UTS0{k}' for k in range(1, 9)],
    'subject': [f'UTS0{k}' for k in range(1, 4)],
}

params_coupled_dict = {
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, 'experiments', '01_calc_resp_decomp.py')

s = {
    'amlt_file': join(repo_dir, 'scripts', 'launch.yaml'),
    # 'sku': '8C7',
    'sku': '8C15',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
    'target___name': 'msrresrchvc',
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    # amlt_kwargs=amlt_kwargs_cpu,
    n_cpus=3,
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
