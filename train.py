from ml_collections import config_flags, config_dict
from absl import app, flags
import sys
import os
from pathlib import Path
from typing import Sequence, Callable
import yaml

import numpy as np
import jax

import logging
logger = logging.getLogger(f'fr.{__name__}')
logger.propagate = False
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter('%(name)s.%(funcName)s:%(lineno)d [%(levelname)s] %(message)s'))
logger.addHandler(_handler)

import wandb
from flowrec._typing import *
from flowrec.training_and_states import save_trainingstate, TrainingState


FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file('config','config.py')
cfg = FLAGS.config
datacfg = FLAGS.config.data_config
casecfg = FLAGS.config.case
mdlcfg = FLAGS.config.model_config
traincfg = FLAGS.config.train_config
wandbcfg = FLAGS.config.wandb_config

WANDB = None
OFFLINE_WANDB = None
DEBUG = None
gpu_id = None
gpu_mem = None
result_dir = './local_results/'
save_dir_name = Path('somewhere')



def fit(
    x_train_batched:Sequence[jax.Array],
    y_train_batched:Sequence[jax.Array],
    x_val:jax.Array,
    y_val:jax.Array,
    state:TrainingState,
    epochs:int,
    rng:jax.random.PRNGKey,
    n_batch:int,
    update:Callable,
    mdl_validation_loss:Callable
):
    '''Train a network'''

    loss_train = []
    loss_val = []
    loss_div = []
    loss_momentum = []
    loss_sensors = []

    best_state = state
    min_loss = np.inf

    for i in range(epochs+1):
        [rng] = jax.random.split(rng,1)

        loss_epoch = []
        loss_epoch_div = []
        loss_epoch_mom = []
        loss_epoch_s = []
        for b in range(n_batch):
            (l, (l_div, l_mom, l_s)), state = update(state, rng, x_train_batched[b], y_train_batched[b])
            if mdlcfg.dropout_rate is None:
                loss_epoch.append(l)
                loss_epoch_div.append(l_div)
                loss_epoch_mom.append(l_mom)
                loss_epoch_s.append(l_s)
            else:
                l, (l_div, l_mom, l_s) = mdl_validation_loss(state.params,None,x_train_batched[b],y_train_batched[b])
                loss_epoch.append(l)
                loss_epoch_div.append(l_div)
                loss_epoch_mom.append(l_mom)
                loss_epoch_s.append(l_s)
            logger.debug(f'batch size is {x_train_batched[b].shape[0]}')
            logger.debug(f'batch: {b}, loss: {l:.7f}.')
        loss_train.append(np.mean(loss_epoch))
        loss_div.append(np.mean(loss_epoch_div))
        loss_momentum.append(np.mean(loss_epoch_mom))
        loss_sensors.append(np.mean(loss_epoch_s))

        
        l_val, _ = mdl_validation_loss(state.params,None,x_val,y_val)
        loss_val.append(l_val)


        if WANDB:
            wandb.log({'loss':loss_train[-1], 'loss_val':l_val, 'loss_div':loss_div[-1], 'loss_momentum':loss_momentum[-1], 'loss_sensors':loss_sensors[-1]})

        if l_val < min_loss:
            best_state = state
            min_loss = l_val
            save_trainingstate(Path(result_dir,save_dir_name),state,'state')

        if i%200 == 0:
            print(f'Epoch: {i}, loss: {loss_train[-1]:.7f}, validation_loss: {l_val:.7f}', flush=True)
            print(f'For training, loss_div: {loss_div[-1]:.7f}, loss_momentum: {loss_momentum[-1]:.7f}, loss_sensors: {loss_sensors[-1]:.7f}')
    state = best_state

    return state, loss_train, loss_val, (loss_div, loss_momentum, loss_sensors)





def save_config(config:config_dict.ConfigDict):
    '''Save config to file config.yml. Load with yaml.unsafe_load(file)'''
    fname = Path(result_dir,save_dir_name,'config.yml')
    with open(fname,'w') as f:
        config_yml = config.to_yaml(stream=f, default_flow_style=False)





def save_results(config:config_dict.ConfigDict):
    pass

def wandb_init(wandb_config):
    pass






def main(_):
    data, datainfo = config.case.load_data(data_config)
    take_observation, insert_observation = config.case.observe(data_config)
    prep_data, make_model = config.select_model(data_config,train_config,model_config)
    observed_train = take_observation(data['u_train'])
    observed_val = take_observation(data['u_val'])
    data.update({
        'y_train':observed_train,
        'y_val':observed_val
    })

    data = prep_data(data)
    x_batched = batch(data['inn_train'])
    y_batched = batch(data['y_train'])

    mdl = make_model(model_config)
    optimiser = config.optimiser(training_config)

    loss_fn = make_loss_fn(
        config,
        take_observation=take_observation,
        insert_observation=insert_observation,
        datainfo = datainfo
    )
    loss_fn_validate = partial(loss_fn)
    update = generate_update_fn(loss_fn)

    if WANDB:
        wandb_init(config)

    fit(train_config)

    if WANDB:
        pass

    save_results()
    save_config()