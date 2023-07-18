import multiprocessing as mp
import jax
import wandb
import subprocess

from argparse import ArgumentParser
from pathlib import Path
from functools import partial

queue = mp.Queue()

def job(randseed:int, experiment:str, save_to:str, epochs:int, prefix:str):

    _experiment = dict([x.split('@') for x in experiment.split(',')])
    
    folder_name = prefix + '-' + str(randseed)

    gpu_id = queue.get()
    
    results_dir = Path(save_to,folder_name)
    results_dir.mkdir()
    log = Path(results_dir,'log')
    err = Path(results_dir,'err')

    command = f"python train.py --gpu_id {gpu_id} --result_dir {save_to} --result_folder_name {folder_name} --wandb --wandbcfg.mode=offline --wandbcfg.group={_experiment['objective']} --cfg train_config/config_experiments.py:{experiment} --cfg.train_config.randseed={randseed} --cfg.train_config.epochs={epochs} --chatty"

    print('Running command: ')
    print(command)
    
    try:
        with open(log,'x') as out, open(err,'x') as error:
            subprocess.run(command.split(" "), stderr=error, stdout=out)
    finally:
        queue.put(gpu_id)





def main(args):

    randseeds = args.randseeds
    save_path = Path(args.save_to)
    if save_path.exists():
        raise ValueError('Experiment path exist, do not overwrite.')
    save_path.mkdir()

    wandb.require("service") # set up wandb for multiprocessing
    wandb.setup()

    for i in range(len(jax.devices())):
        queue.put(i)
    
    run_job = partial(job, experiment=args.experiment, save_to=args.save_to, epochs=args.epochs, prefix=args.job_prefix)

    pool = mp.Pool(len(jax.devices())) # one job per device
    pool.map(run_job,randseeds)

    pool.close()
    pool.join()

    


if __name__ == '__main__':


    parser = ArgumentParser(description='Run repeats of the same experiments.')
    parser.add_argument('--experiment', help="The config str to identify this experiment. This is passed to config_experiments.py", required=True)
    parser.add_argument('--save_to', help='path to the result folder', required=True)
    parser.add_argument('--randseeds', type=int, nargs='+',help='a list of random seeds for weight initialisation', required=True)
    parser.add_argument('--job_prefix', help='Prefix for each run', required=True)
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs per run.')
    args = parser.parse_args()


    main(args)