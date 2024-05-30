import multiprocessing as mp
import wandb
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from functools import partial

case_command = {
    'physicsreplacemean': "--cfg.train_config.lr_scheduler=cyclic_decay_default --wandbcfg.tags=('task:minimum_observation','milestone:2dkol') --cfg.model_config.b1_channels=(4,) --cfg.model_config.b1_filters=((5,5),) --cfg.model_config.b2_filters=((5,5),) --cfg.model_config.fft_branch=false --cfg.train_config.learning_rate=0.0047 --cfg.train_config.nb_batches=10 --cfg.train_config.weight_sensors=64.", 
    'physicswithdata': "--cfg.train_config.lr_scheduler=cyclic_decay_default --wandbcfg.tags=('task:minimum_observation','milestone:2dkol') --cfg.model_config.b1_channels=(1,) --cfg.model_config.b1_filters=((5,5),) --cfg.model_config.b2_filters=((5,5,),) --cfg.model_config.fft_branch=true --cfg.train_config.learning_rate=0.0008 --cfg.train_config.nb_batches=38",
    'physicsnoreplace': "--cfg.train_config.lr_scheduler=cyclic_decay_default --wandbcfg.tags=('task:minimum_observation','milestone:2dkol') --cfg.model_config.b1_channels=(4,4) --cfg.model_config.b1_filters=((5,5),) --cfg.model_config.b2_filters=((5,5),) --cfg.model_config.fft_branch=false --cfg.train_config.learning_rate=0.004 --cfg.train_config.nb_batches=33 --cfg.train_config.weight_sensors=8.", 
}
datapath = './local_data/kolmogorov/dim2_re34_k32_f4_dt1_grid128_14635.h5'

queue = mp.Queue()
EXCEPTION_COUNT = 0
_randseed_input = 3612
_randseed_sensors = 123489



def job_reduce_input(save_to:str, epochs:int, loss_fn:str, num_sensors:int, num_input:int):
    folder_name = f'min_input-{num_input}-{num_sensors}'
    gpu_id = queue.get()
    print(Path(datapath),Path(datapath).exists())

    result_dir = Path(save_to, folder_name)
    result_dir.mkdir()
    base_command = f"python train.py --gpu_id {gpu_id} --result_dir {save_to} --result_folder_name {folder_name} --cfg train_config/config.py:observe@random_pin,dataloader@2dkol,loss_fn@{loss_fn} --wandb --wandbcfg train_config/config_wandb.py:observe@random_pin,dataloader@2dkol,loss_fn@{loss_fn} --wandbcfg.mode=offline --cfg.data_config.random_sensors=({_randseed_sensors},{num_sensors}) --cfg.data_config.random_input=({_randseed_input},{num_input}) --cfg.train_config.epochs={epochs} --cfg.data_config.data_dir={datapath}"
    command = base_command + " " + case_command[loss_fn]

    print('Running command:   ')
    print(command)

    log = Path(result_dir, 'log')
    err = Path(result_dir, 'err')

    try:
        with open(log,'x') as out, open(err,'x') as error:
            subprocess.run(command.split(" "), stderr=error, stdout=out)
    except Exception as e:
        # Handle all other exceptions
        global EXCEPTION_COUNT
        print(f"An error occurred: {e}")
        EXCEPTION_COUNT = EXCEPTION_COUNT + 1
        if EXCEPTION_COUNT >= 2:
            raise ValueError('Program failed twice.')
    finally:
        queue.put(gpu_id)


def job_reduce_sensors(save_to:str, epochs:int, loss_fn:str, num_input:int, num_sensors:int):
    folder_name = f'min_sensors-{num_input}-{num_sensors}'
    gpu_id = queue.get()

    result_dir = Path(save_to, folder_name)
    result_dir.mkdir()
    base_command = f"python train.py --gpu_id {gpu_id} --result_dir {save_to} --result_folder_name {folder_name} --cfg train_config/config.py:observe@random_pin,dataloader@2dkol,loss_fn@{loss_fn} --wandb --wandbcfg train_config/config_wandb.py:observe@random_pin,dataloader@2dkol,loss_fn@{loss_fn} --wandbcfg.mode=offline --cfg.data_config.random_sensors=({_randseed_sensors},{num_sensors}) --cfg.data_config.random_input=({_randseed_input},{num_input}) --cfg.train_config.epochs={epochs} --cfg.data_config.data_dir={datapath}"
    command = base_command + " " + case_command[loss_fn]

    print('Running command:   ')
    print(command)

    log = Path(result_dir, 'log')
    err = Path(result_dir, 'err')
    try:
        with open(log,'x') as out, open(err,'x') as error:
            subprocess.run(command.split(" "), stderr=error, stdout=out)
    except Exception as e:
        # Handle all other exceptions
        global EXCEPTION_COUNT
        print(f"An error occurred: {e}")
        EXCEPTION_COUNT = EXCEPTION_COUNT + 1
        if EXCEPTION_COUNT >= 2:
            raise ValueError('Program failed twice.')
    finally:
        queue.put(gpu_id)



def main(args):
    jobtype = args.jobtype
    save_path = Path(args.save_to)
    save_path.mkdir(exist_ok=True)

    wandb.require("service")
    wandb.setup()

    for i in range(args.numgpu):
        queue.put(i)
    
    if jobtype == 'job_input':
        print('Running cases with different number of input.')
        num_input = args.num_input
        run_job = partial(
            job_reduce_input, 
            save_path, 
            args.epochs, 
            args.loss_fn,
            args.num_sensors
        )
        
        pool = mp.Pool(args.numgpu)
        pool.map(run_job, num_input)

        pool.close()
        pool.join()
    elif jobtype == 'job_sensors':
        print('Running cases with different number of sensors.')
        num_sensors = args.num_sensors
        run_job = partial(
            job_reduce_sensors, 
            save_path, 
            args.epochs, 
            args.loss_fn,
            args.num_input
        )
        
        pool = mp.Pool(args.numgpu)
        pool.map(run_job, num_sensors)

        pool.close()
        pool.join()
    else:
        print("jobtype must be either 'job_input' or 'job_sensors'.")
    



if __name__ == '__main__':
    parser = ArgumentParser(prog = 'Reduce input and normal sensors.')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs per run.')
    parser.add_argument('--numgpu', type=int, help='Number of gpus available.', required=True)
    parser.add_argument('--save_to', help='path to the result folder', required=True)
    parser.add_argument('--loss_fn', help='Which loss function to use?', required=True)
    
    # add subparsers
    subparsers = parser.add_subparsers(required=True,dest='jobtype')

    parser1 = subparsers.add_parser('job_input', help='Run cases with different number of inputs.')
    parser1.add_argument('--num_input', type=int, nargs='+',help='A list of number of input sensors to use.', required=True)
    parser1.add_argument('--num_sensors', type=int, help='Number of regular sensors', required=True)
    
    parser2 = subparsers.add_parser('job_sensors', help='Run cases with different numbers of regular sensors in the domain.')
    parser2.add_argument('--num_sensors', type=int, nargs='+',help='A list of number of regular sensors to use.', required=True)
    parser2.add_argument('--num_input', type=int, help='Number of inputs.', required=True)

    args=parser.parse_args()
    main(args)
