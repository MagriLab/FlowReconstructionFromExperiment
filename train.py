from ml_collections import config_flags, config_dict
from absl import app, flags
import sys
import os
from pathlib import Path
from typing import Sequence, Callable
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

import logging
logger = logging.getLogger(f'fr.{__name__}')
logger.propagate = False
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter('%(name)s.%(funcName)s:%(lineno)d [%(levelname)s] %(message)s'))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

import wandb
from flowrec._typing import *
from flowrec.training_and_states import save_trainingstate, TrainingState, generate_update_fn
from utils.py_helper import update_matching_keys


FLAGS = flags.FLAGS


print("Started at: ", time.asctime(time.localtime(time.time())))
time_stamp = time.strftime("%y%m%d%H%M%S",time.localtime(time.time()))

# ====================== system config ============================
flags.DEFINE_bool('wandb',False,'Use --wandb to log the experiment to wandb.')
flags.DEFINE_multi_string('debug',None,'Run these scripts in debug mode.')
flags.DEFINE_integer('gpu_id',0,'Which gpu use.')
flags.DEFINE_float('gpu_mem',0.3,'Fraction of gpu memory to use.')
flags.DEFINE_string('result_dir','./local_results/','Path to a directory where the result will be saved.')
flags.DEFINE_string('result_folder_name',str(time_stamp),'Name of the folder where all files from this run will save to. Default the time stamp.')
flags.DEFINE_bool('chatty',True,'Print information on where the program is at now.')


# ======================= test config ===============================
_CONFIG = config_flags.DEFINE_config_file('cfg','config.py')
_WANDB = config_flags.DEFINE_config_file('wandbcfg','config_wandb.py','path to wandb config file.')



def debugger(loggers):
    for l in loggers:
        logging.getLogger(f'fr.{l}').setLevel(logging.DEBUG)



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
    mdl_validation_loss:Callable,
    wandb_run
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
            if FLAGS.cfg.model_config.dropout_rate is None:
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


        if FLAGS.wandb:
            wandb_run.log({'loss':loss_train[-1], 'loss_val':l_val, 'loss_div':loss_div[-1], 'loss_momentum':loss_momentum[-1], 'loss_sensors':loss_sensors[-1]})

        if l_val < min_loss:
            best_state = state
            min_loss = l_val
            save_trainingstate(Path(FLAGS.result_dir,FLAGS.result_folder_name),state,'state')

        if i%200 == 0:
            print(f'Epoch: {i}, loss: {loss_train[-1]:.7f}, validation_loss: {l_val:.7f}', flush=True)
            print(f'For training, loss_div: {loss_div[-1]:.7f}, loss_momentum: {loss_momentum[-1]:.7f}, loss_sensors: {loss_sensors[-1]:.7f}')
    state = best_state

    return state, loss_train, loss_val, (loss_div, loss_momentum, loss_sensors)


def batching(nb_batches:int, data:jax.Array):
    '''Split data into nb_batches number of batches along axis 0.'''
    return jnp.array_split(data,nb_batches,axis=0)



def save_config(config:config_dict.ConfigDict):
    '''Save config to file config.yml. Load with yaml.unsafe_load(file)'''
    fname = Path(FLAGS.result_dir,FLAGS.result_folder_name,'config.yml')
    with open(fname,'x') as f:
        config.to_yaml(stream=f, default_flow_style=False)



def save_results(config:config_dict.ConfigDict):
    pass



def wandb_init(wandbcfg:config_dict.ConfigDict):
    run = wandb.init(**wandbcfg)

    if wandbcfg.config.weight_physcis > 0.0:
        run.tags = run.tags + ('PhysicsInformed',)
    if wandbcfg.config.weight_sensors == 0:
        run.tags = run.tags + ('PhysicsOnly',)
    
    return run





def main(_):

    cfg = FLAGS.cfg
    datacfg = FLAGS.cfg.data_config
    mdlcfg = FLAGS.cfg.model_config
    traincfg = FLAGS.cfg.train_config
    wandbcfg = FLAGS.wandbcfg
    logger.info(f'List of options selected from optional functions: \n      {cfg.case.values()}')

    # ===================== setting up system ==========================
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(FLAGS.gpu_mem)

    if FLAGS.chatty:
        logger.setLevel(logging.INFO)
    
    if FLAGS.debug:
        debugger(FLAGS.debug)
        logger.info(f'Running these scripts in debug mode: {FLAGS.debug}.')
    

    # =================== pre-processing ================================
    
    # data has u_train, u_val, inn_train, inn_val [t, space..., 3] or [t, len]
    logger.info('Loading data.')
    data, datainfo = cfg.case.dataloader(datacfg)
    logger.debug(f'Data dictionary has {data.keys()}')
    logger.debug(f'Datainfo is {datainfo}')

    logger.info('Taking observations.')
    take_observation, insert_observation = cfg.case.observe(datacfg)
    observed_train = take_observation(data['u_train'])
    observed_val = take_observation(data['u_val'])
    
    data.update({
        'y_train':observed_train,
        'y_val':observed_val
    })
    logger.debug(f'Data dict now has {data.keys()}')

    percent_observed = 100*(observed_train[0,...,0].size/data['u_train'][0,...,0].size)

    # ==================== set up model ==============================
    rng = jax.random.PRNGKey(time_stamp)
    optimizer = optax.adamw(
        learning_rate=traincfg.learning_rate,
        weight_decay=traincfg.regularisation_strength
    )

    prep_data, make_model = cfg.case.select_model(datacfg,mdlcfg,traincfg)
    logger.info('Selected a model.')
    
    if FLAGS.wandb:
        logger.info('Updating wandb config with experiment config')
        update_matching_keys(wandbcfg.config, datacfg)
        update_matching_keys(wandbcfg.config, mdlcfg)
        update_matching_keys(wandbcfg.config, traincfg)
        update_matching_keys(wandbcfg.config, {'percent_observed':percent_observed})
        run = wandb_init(wandbcfg)
        logger.info('Successfully initalised werights and biases.')


    mdl = make_model(mdlcfg)
    logger.info('Made a model.')

    params = mdl.init(rng,data['inn_val'][0,...])
    logger.info('Initialised weights.')
    logger.debug(jax.tree_util.tree_map(lambda x: x.shape,params))
    
    opt_state = optimizer.init(params)
    logger.info('Initialised optimiser.')
    
    state = TrainingState(params, opt_state)


    # =================== loss function ==========================

    loss_fn = cfg.case.loss_fn(
        cfg,
        datainfo = datainfo,
        take_observation = take_observation,
        insert_observation = insert_observation
    )
    logger.info('Created loss function.')
    mdl_validation_loss = jax.jit(jax.tree_util.Partial(loss_fn,mdl.apply,TRAINING=False))
    update = generate_update_fn(mdl.apply,optimizer,loss_fn,kwargs_value_and_grad={'has_aux':True}) # this update weights once.


    # ==================== start training ===========================

    data = prep_data(data)
    x_batched = batching(traincfg.nb_batches, data['inn_train'])
    y_batched = batching(traincfg.nb_batched, data['y_train'])
    logger.info('Prepared data as required by the model selected and batched the data.')


    logger.info('Starting training now...')
    state, loss_train, loss_val, (loss_div, loss_momentum, loss_sensors) = fit(
        x_train_batched=x_batched,
        y_train_batched=y_batched,
        x_val=data['inn_val'],
        y_val=data['y_val'],
        state=state,
        epochs=traincfg.epoches,
        rng=rng,
        n_batch=traincfg.nb_batches,
        update=update,
        mdl_validation_loss=mdl_validation_loss,
        wandb_run=run
    )


    if FLAGS.wandb:
        run.finish()
    logger.info('Finished training.')

    logger.info(f'writing configuration and results to {FLAGS.result_folder_name}')
    save_config(cfg)
    save_results(cfg)





if __name__ == '__main__':

    app.run(main)

    print("Finished at: ", time.asctime(time.localtime(time.time())))