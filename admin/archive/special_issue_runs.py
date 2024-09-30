import time
import os
import wandb
import subprocess
import itertools as it
import multiprocessing as mp

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

queue = mp.Queue()
command_list_start = [
    "python",
    "train.py",
    "--wandb",
    "--gpu_mem=1",
]
shared_config_values = [
    "--wandbcfg.mode=offline",
    "--cfg.data_config.data_dir=./local_data/kolmogorov/dim2_re40_k32_dt1_T800_grid128_586178_short.h5",
    "--cfg.data_config.train_test_split=(6000,200,200)",
    "--cfg.data_config.randseed=163571", 
    "--cfg.data_config.snr=10",
    '--cfg.data_config.random_sensors=(16,50)',
    '--cfg.data_config.random_input=(28,50)',
    '--cfg.model_config.b1_channels=(1,)',
    '--cfg.data_config.normalise=False', 
    '--cfg.train_config.nb_batches=30',
    '--cfg.data_config.re=40',
]

# =================== options ==========================
options_loss3 = {
    'model_config.dropout_rate':[0.0, 0.001],
    'train_config.learning_rate':[0.001, 0.003, 0.005],
    'train_config.regularisation_strength':[0.0005, 0.002, 0.004],
    'train_config.lr_scheduler':['cyclic_decay_default'],
    'model_config.fft_branch':['False','True'],
    'model_config.b2_filters':[((5,5),)]
}
def grid_search_loss3(config_values, save_path, epochs, tags, avoid=None):
    i = config_values.pop('id')
    if avoid is not None:
        if i in avoid:
            print(f'User instruct to not run case {i}.')
            return
    print('id', i)
    folder_name = f'gridsearch-{i}'
    results_dir = Path(save_path,folder_name)
    gpu_id = queue.get()
    command_list = command_list_start.copy()
    command_list.extend([
        f'--gpu_id={gpu_id}',
        f'--result_dir={save_path}',
        f'--result_folder_name={folder_name}',
        '--cfg=train_config/config.py:observe@random_pin,dataloader@2dkol,loss_fn@physicswithdata',
        '--wandbcfg=train_config/config_wandb.py:observe@random_pin,dataloader@2dkol,loss_fn@physicswithdata',
        f'--cfg.train_config.epochs={epochs}',
        f"--wandbcfg.tags={tags}"
    ])
    command_list.extend(shared_config_values)
    new_values = [f'--cfg.{key}={item}' for key,item in config_values.items()]
    command_list.extend(new_values)
    print('Running command: ', flush=True)
    print(" ".join(command_list), flush=True)
    results_dir.mkdir()
    log = Path(results_dir,'log')
    err = Path(results_dir,'err')
    try:
        with open(log,'x') as out, open(err,'x') as error:
            subprocess.run(command_list, stderr=error, stdout=out)
    finally:
        queue.put(gpu_id)



options_loss1 = {
    'model_config.dropout_rate':[0.0005, 0.001],
    'train_config.learning_rate':[0.0005, 0.001, 0.002],
    'train_config.regularisation_strength':[0.002, 0.005],
    'train_config.lr_scheduler':['cyclic_decay_default'],
    'train_config.weight_sensors': [5.0, 10.0, 20.0],
    'model_config.fft_branch':['False'],
    'model_config.b2_filters':[((5,5),)],
}
def grid_search_loss1(config_values, save_path, epochs, tags, avoid=None):
    i = config_values.pop('id')
    if avoid is not None:
        if i in avoid:
            print(f'User instruct to not run case {i}.')
            return
    folder_name = f'gridsearch-{i}'
    results_dir = Path(save_path,folder_name)
    gpu_id = queue.get()
    command_list = command_list_start.copy()
    command_list.extend([
        f'--gpu_id={gpu_id}',
        f'--result_dir={save_path}',
        f'--result_folder_name={folder_name}',
        '--cfg=train_config/config.py:observe@random_pin,dataloader@2dkol,loss_fn@physicsnoreplace',
        '--wandbcfg=train_config/config_wandb.py:observe@random_pin,dataloader@2dkol,loss_fn@physicsnoreplace',
        f'--cfg.train_config.epochs={epochs}',
        f'--wandbcfg.tags=({tags})'
    ])
    command_list.extend(shared_config_values)
    new_values = [f'--cfg.{key}={item}' for key,item in config_values.items()]
    command_list.extend(new_values)
    print('Running command: ')
    print(" ".join(command_list))
    results_dir.mkdir()
    log = Path(results_dir,'log')
    err = Path(results_dir,'err')
    try:
        with open(log,'x') as out, open(err,'x') as error:
            subprocess.run(command_list, stderr=error, stdout=out)
    finally:
        queue.put(gpu_id)



# ======================================
def main(*args, avoid=None):
    job, options, save_path, epochs, tags = args
    if not save_path.exists():
        print(f'Creating a new experiment directory {save_path}')
        save_path.mkdir()
    
    wandb.require("service") # set up wandb for multiprocessing
    wandb.setup()

    gpustr = os.environ['CUDA_VISIBLE_DEVICES']
    gpulst = gpustr.split(",")
    for gpu_id in gpulst:
        queue.put(int(gpu_id))
    
    run_job = partial(job, save_path=save_path, epochs=epochs, tags=tags, avoid=avoid)

    keys, values = zip(*options.items())
    config_values = [dict(zip(keys, p)) for p in it.product(*values)]
    for i,d in enumerate(config_values):
        d['id'] = i

    num_devices = len(gpulst)
    pool = mp.Pool(num_devices) # one job per device
    pool.map(run_job,config_values)

    pool.close()
    pool.join()


if __name__ == '__main__':

    # =========== change these ================
    job, options = grid_search_loss3, options_loss3
    save_path = Path('./local_results/2dkol/extreme-events/gridsearchloss3-snr10/')
    epochs = 10000
    tags = '("test:special-issue",)'
    
    parser = ArgumentParser(description='Run repeats of the same experiments.')
    parser.add_argument('--avoid', type=int, nargs='+',help="a list of random seeds for placing sensors using the observation option 'random_pin'", required=False)
    args = parser.parse_args()

    main(job, options, save_path, epochs, tags, avoid=args.avoid)
