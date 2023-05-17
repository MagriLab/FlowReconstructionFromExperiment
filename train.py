from ml_collections import config_flags, config_dict
from absl import app, flags
import sys
import os
import re
from pathlib import Path
from typing import Sequence, Callable
import time
import h5py

import numpy as np
import jax
import jax.numpy as jnp
import optax

import wandb
from flowrec._typing import *
from flowrec.training_and_states import save_trainingstate, TrainingState, generate_update_fn
from utils.py_helper import update_matching_keys
from utils.system import temporary_fix_absl_logging
from train_config.sweep_process_config import sweep_preprocess_cfg
from train_config.option_codes import code

import logging
logger = logging.getLogger(f'fr.{__name__}')
logger.propagate = False
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter('%(name)s.%(funcName)s:%(lineno)d [%(levelname)s] %(message)s'))
logger.addHandler(_handler)
logger.setLevel(logging.WARNING)

temporary_fix_absl_logging()



FLAGS = flags.FLAGS


print("Started at: ", time.asctime(time.localtime(time.time())))
time_stamp = time.strftime("%y%m%d%H%M%S",time.localtime(time.time()))

# ====================== system config ============================
flags.DEFINE_bool('wandb',False,'Use --wandb to log the experiment to wandb.')
flags.DEFINE_bool('wandb_sweep',False,'Run script in wandb sweep mode.')
flags.DEFINE_multi_string('debug',None,'Run these scripts in debug mode.')
flags.DEFINE_integer('gpu_id',None,'Which gpu use.')
flags.DEFINE_float('gpu_mem',0.3,'Fraction of gpu memory to use.')
flags.DEFINE_string('result_dir','./local_results/','Path to a directory where the result will be saved.')
flags.DEFINE_string('result_folder_name',None,'Name of the folder where all files from this run will save to. Default the time stamp.')
flags.DEFINE_bool('chatty',False,'Print information on where the program is at now.')


# ======================= test config ===============================
_CONFIG = config_flags.DEFINE_config_file('cfg','train_config/config.py')
_WANDB = config_flags.DEFINE_config_file('wandbcfg','train_config/config_wandb.py','path to wandb config file.')



def debugger(loggers:Sequence[str]):
    '''Set the logging level to DEBUG for the list of loggers. 
    The input is a list of the names of the loggers.'''
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
    tmp_dir:Path,
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
            
            if b == 0 or b == n_batch-1:
                logger.debug(f'batch {b} has size {x_train_batched[b].shape[0]}, loss: {l:.7f}.')

        loss_train.append(np.mean(loss_epoch))
        loss_div.append(np.mean(loss_epoch_div))
        loss_momentum.append(np.mean(loss_epoch_mom))
        loss_sensors.append(np.mean(loss_epoch_s))

        
        l_val, _ = mdl_validation_loss(state.params,None,x_val,y_val)
        loss_val.append(l_val)


        if FLAGS.wandb:
            logger.debug('Logging with wandb.')
            wandb_run.log({'loss':loss_train[-1], 'loss_val':float(l_val), 'loss_div':loss_div[-1], 'loss_momentum':loss_momentum[-1], 'loss_sensors':loss_sensors[-1]})

        if l_val < min_loss:
            best_state = state
            min_loss = l_val
            save_trainingstate(tmp_dir,state,'state')

        if i%200 == 0:
            print(f'Epoch: {i}, loss: {loss_train[-1]:.7f}, validation_loss: {l_val:.7f}', flush=True)
            print(f'For training, loss_div: {loss_div[-1]:.7f}, loss_momentum: {loss_momentum[-1]:.7f}, loss_sensors: {loss_sensors[-1]:.7f}')
    state = best_state

    return state, loss_train, loss_val, (loss_div, loss_momentum, loss_sensors)


def batching(nb_batches:int, data:jax.Array):
    '''Split data into nb_batches number of batches along axis 0.'''
    return jnp.array_split(data,nb_batches,axis=0)



def save_config(config:config_dict.ConfigDict, tmp_dir:Path):
    '''Save config to file config.yml. Load with yaml.unsafe_load(file)'''
    fname = Path(tmp_dir,'config.yml')
    with open(fname,'x') as f:
        config.to_yaml(stream=f, default_flow_style=False)





def wandb_init(wandbcfg:config_dict.ConfigDict):

    if not wandbcfg.name:
        wandbcfg.update({'name':FLAGS.result_folder_name})

    cfg_dict = wandbcfg.to_dict()
    logger.debug(f'Arguments passed to wandb.init {cfg_dict}.')

    run = wandb.init(**cfg_dict)

    _pattern = re.compile(".*/FlowReconstructionFromExperiment/local")
    def _ignore_files(path):
        return _pattern.match(path)
    
    run.log_code('.', exclude_fn=_ignore_files)
    
    return run





def main(_):

    cfg = FLAGS.cfg
    datacfg = FLAGS.cfg.data_config
    mdlcfg = FLAGS.cfg.model_config
    traincfg = FLAGS.cfg.train_config
    wandbcfg = FLAGS.wandbcfg
    logger.info(f'List of options selected from optional functions: \n      {cfg.case.values()}')

    # ===================== setting up system ==========================
    if FLAGS.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(FLAGS.gpu_mem)
    
    if not FLAGS.result_folder_name and not FLAGS.wandb_sweep:
        _folder = code(cfg.case)
        _folder = _folder + str(time_stamp)
        FLAGS.result_folder_name = _folder

    tmp_dir = Path(FLAGS.result_dir,FLAGS.result_folder_name)

    if FLAGS.chatty:
        logger.setLevel(logging.INFO)
    
    if FLAGS.debug:
        debugger(FLAGS.debug)
        logger.info(f'Running these scripts in debug mode: {FLAGS.debug}.')
    
    if FLAGS.wandb_sweep:
        FLAGS.wandb = True

    # =================== pre-processing ================================
    
    # data has u_train, u_val, inn_train, inn_val [t, space..., 3] or [t, len]
    logger.info('Loading data.')
    data, datainfo = cfg.case.dataloader(datacfg)
    logger.debug(f'Data dictionary has {data.keys()}')
    logger.debug(f'Datainfo is {datainfo}')

    logger.info('Taking observations.')
    take_observation, insert_observation = cfg.case.observe(
        datacfg,
        example_pred_snapshot = data['u_train'][0,...],
        example_pin_snapshot = data['inn_train'][0,...]
    )
    observed_train = take_observation(data['u_train'])
    observed_val = take_observation(data['u_val'])
    
    data.update({
        'y_train':observed_train,
        'y_val':observed_val
    })
    logger.debug(f'Data dict now has {data.keys()}')

    percent_observed = 100*(observed_train.size/data['u_train'].size)

    # ==================== set up model ==============================
    rng = jax.random.PRNGKey(int(time_stamp))
    optimizer = optax.adamw(
        learning_rate=traincfg.learning_rate,
        weight_decay=traincfg.regularisation_strength
    )

    prep_data, make_model = cfg.case.select_model(datacfg = datacfg, mdlcfg = mdlcfg, traincfg = traincfg)
    logger.info('Selected a model.')
    
    ## Initialise wandb
    if FLAGS.wandb:
        logger.info('Updating wandb config with experiment config')
        update_matching_keys(wandbcfg.config, datacfg)
        update_matching_keys(wandbcfg.config, mdlcfg)
        update_matching_keys(wandbcfg.config, traincfg)
        update_matching_keys(wandbcfg.config, cfg.case)
        update_matching_keys(wandbcfg.config, {'percent_observed':percent_observed})
        run = wandb_init(wandbcfg)
        # wandb.config.update({'percent_observed':percent_observed})
        logger.info('Successfully initialised weights and biases.')

        if FLAGS.wandb_sweep:
            sweep_params = sweep_preprocess_cfg(wandb.config)
            update_matching_keys(datacfg, sweep_params)
            update_matching_keys(mdlcfg, sweep_params)
            update_matching_keys(traincfg, sweep_params)
            logger.info('Running in sweep mode, replace config parameters with sweep parameters.')
            logger.debug(f'Running with {sweep_params}')

    else:
        run = None


    logger.debug(f'The finalised data dictionary has items {data.keys()}')
    data = prep_data(data)
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

    mdl_validation_loss = jax.jit(
        jax.tree_util.Partial(
            loss_fn,
            mdl.apply,
            apply_kwargs={'TRAINING':False},
            y_minmax=data['val_minmax']
        )
    )
    update = generate_update_fn(
        mdl.apply,
        optimizer,
        loss_fn,
        kwargs_value_and_grad={'has_aux':True}, 
        kwargs_loss={'y_minmax':data['train_minmax']}
    ) # this update weights once.


    # ==================== start training ===========================

    x_batched = batching(traincfg.nb_batches, data['inn_train'])
    y_batched = batching(traincfg.nb_batches, data['y_train'])
    logger.info('Prepared data as required by the model selected and batched the data.')
    logger.debug(f'First batch of input data has shape {x_batched[0].shape}.')


    logger.info('Starting training now...')
    state, loss_train, loss_val, (loss_div, loss_momentum, loss_sensors) = fit(
        x_train_batched=x_batched,
        y_train_batched=y_batched,
        x_val=data['inn_val'],
        y_val=data['y_val'],
        state=state,
        epochs=traincfg.epochs,
        rng=rng,
        n_batch=traincfg.nb_batches,
        update=update,
        mdl_validation_loss=mdl_validation_loss,
        tmp_dir=tmp_dir,
        wandb_run=run
    )


    logger.info('Finished training.')

    # ===================== Save results
    logger.info(f'writing configuration and results to {FLAGS.result_folder_name}')
    save_config(cfg,tmp_dir)

    with h5py.File(Path(tmp_dir,'results.h5'),'w') as hf:
        hf.create_dataset("loss_train",data=np.array(loss_train))
        hf.create_dataset("loss_val",data=np.array(loss_val))
        hf.create_dataset("loss_div",data=np.array(loss_div))
        hf.create_dataset("loss_momentum",data=np.array(loss_momentum))
        hf.create_dataset("loss_sensors",data=np.array(loss_sensors))
    

    # ============= Save only best model if doing sweep ==========
    if FLAGS.wandb_sweep:
        api = wandb.Api()

        best_run = api.sweep(f'{run.entity}/{run.project}/{run.sweep_id}').best_run()
        
        try: 
            if loss_train[-1] < best_run.summary['loss']:
                logger.info('Best model so far, saving weights and configurations.')
                artifact = wandb.Artifact(name=f'sweep_weights_{run.sweep_id}', type='model') 
                artifact.add_dir(tmp_dir)
                run.log_artifact(artifact)
                # run.finish_artifact(artifact) # only necessary for distributed runs
        except KeyError as e: # probably the first run of the sweep
            logger.warning(e)
            artifact = wandb.Artifact(name=f'sweep_weights_{run.sweep_id}', type='model') 
            artifact.add_dir(tmp_dir)
            run.log_artifact(artifact)
            # run.finish_artifact(artifact)
            
        for child in tmp_dir.iterdir(): 
            child.unlink()
        tmp_dir.rmdir()
    
    
    if FLAGS.wandb:
        run.finish()





if __name__ == '__main__':

    app.run(main)

    print("Finished at: ", time.asctime(time.localtime(time.time())))