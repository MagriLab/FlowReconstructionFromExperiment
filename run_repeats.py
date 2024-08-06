import os
import wandb
import subprocess
import multiprocessing as mp

from argparse import ArgumentParser
from pathlib import Path
from functools import partial

queue = mp.Queue()

def job(randseeds:tuple, experiment:str, save_to:str, epochs:int, prefix:str, use_artifact:str=None):
    """randseed = (weight, sensors)"""

    _experiment = dict([x.split('@') for x in experiment.split(',')])
    
    folder_name = prefix + '-' + '-'.join([str(_s) for _s in randseeds if _s is not None])

    rand_w, rand_s = randseeds

    gpu_id = queue.get()
    
    results_dir = Path(save_to,folder_name)
    results_dir.mkdir()
    log = Path(results_dir,'log')
    err = Path(results_dir,'err')
    
    # extra things to pass to experiment config
    if rand_s is not None:
        experiment = experiment + "," + f"sensor_randseed@{rand_s}"
        
    command = f"python train.py --gpu_id {gpu_id} --result_dir {save_to} --result_folder_name {folder_name} --wandb --wandbcfg.mode=offline --wandbcfg.group={_experiment['objective']} --cfg train_config/config_experiments.py:{experiment} --cfg.train_config.randseed={rand_w} --cfg.train_config.epochs={epochs} --chatty"

    if use_artifact is not None:
        command = command + " " + f"--wandbcfg.use_artifact={use_artifact}"

    print('Running command: ')
    print(command)
    
    try:
        with open(log,'x') as out, open(err,'x') as error:
            subprocess.run(command.split(" "), stderr=error, stdout=out)
    finally:
        queue.put(gpu_id)





def main(args):

    weight_seeds = args.randseeds
    if args.sensor_randseeds is None:
        args.sensor_randseeds = [None]*len(weight_seeds)
    randseeds = zip(weight_seeds, args.sensor_randseeds)
    save_path = Path(args.save_to)
    if not save_path.exists():
    #     raise ValueError('Experiment path exist, do not overwrite.')
        print(f'Creating a new experiment directory {save_path}')
        save_path.mkdir()

    wandb.require("service") # set up wandb for multiprocessing
    wandb.setup()

    gpustr = os.environ['CUDA_VISIBLE_DEVICES']
    gpulst = gpustr.split(",")
    for gpu_id in gpulst:
        queue.put(int(gpu_id))
    
    run_job = partial(job, experiment=args.experiment, save_to=args.save_to, epochs=args.epochs, prefix=args.job_prefix, use_artifact=args.use_artifact)

    pool = mp.Pool(len(gpulst)) # one job per device
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
    # parser.add_argument('--numgpu', type=int, help='Number of gpus available.', required=True)
    parser.add_argument('--sensor_randseeds', type=int, nargs='+',help="a list of random seeds for placing sensors using the observation option 'random_pin'", required=False)
    parser.add_argument('--use_artifact', help="name of the wandb artifact to use", required=False)
    args = parser.parse_args()


    main(args)
