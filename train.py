from ml_collections import config_flags, config_dict
from absl import app, flags
import sys
import os
import re
from pathlib import Path
from typing import Sequence, Callable, Optional
from timeit import timeit
import time
import h5py
import warnings
import yaml

import wandb
import jax
import numpy as np
import jax.numpy as jnp

from flowrec._typing import *
from flowrec.training_and_states import save_trainingstate, TrainingState, generate_update_fn, restore_trainingstate, params_merge
from flowrec.losses import loss_mse
from flowrec.utils.py_helper import update_matching_keys
from flowrec.utils.system import temporary_fix_absl_logging, set_gpu
from train_config.sweep_process_config import sweep_preprocess_cfg
from train_config.option_codes import code
from train_config.train_options.optimizer import get_optimizer
from train_config.config_wandb import get_config as get_wandb_config_experiment

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
flags.DEFINE_bool('wandb_sweep',False,'Run script in wandb sweep mode.')
flags.DEFINE_multi_string('debug',None,'Run these scripts in debug mode.')
flags.DEFINE_string('gpu_id',None,'Which gpu use.')
flags.DEFINE_float('gpu_mem',0.9,'Fraction of gpu memory to use.')
flags.DEFINE_string('result_dir','./local_results/3dkol/','Path to a directory where the result will be saved.')
flags.DEFINE_string('result_folder_name',None,'Name of the folder where all files from this run will save to. Default the time stamp.')
flags.DEFINE_bool('chatty',False,'Print information on where the program is at now.')
flags.DEFINE_bool('resume',False, "True for resuming an exisitng model using the same weights")
flags.DEFINE_integer('print', 50, "How many epochs to print losses screen.")


## Define interal global variables
flags.DEFINE_bool('_noisy',False,'DO NOT CHANGE! True loss will be calculated with clean data if the data is noisy.')
flags.DEFINE_string('_experimentcfgstr',None,'DO NOT CHANGE! For use with experiments only.')


# ======================= test config ===============================
_CONFIG = config_flags.DEFINE_config_file('cfg','train_config/config.py')
_WANDB = config_flags.DEFINE_config_file('wandbcfg','train_config/config_wandb.py','path to wandb config file.')



def debugger(loggers:Sequence[str]):
    '''Set the logging level to DEBUG for the list of loggers. 
    The input is a list of the names of the loggers.'''
    for l in loggers:
        logging.getLogger(f'fr.{l}').setLevel(logging.DEBUG)


def save_config(config:config_dict.ConfigDict, tmp_dir:Path):
    '''Save config to file config.yml. Load with yaml.unsafe_load(file)'''
    logger.info(f'save config to {tmp_dir}.')
    fname = Path(tmp_dir,'config.yml')
    with open(fname,'x') as f:
        config.to_yaml(stream=f, default_flow_style=False)


def update_resume_config(resume_from: Path, cfg:config_dict.ConfigDict):
    """Compare the update the old config from previous training with the new user config. train_config is not updated."""
    with open(Path(resume_from,'config.yml'),'r') as f:
        old_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
        logger.info('Successfully loaded old configs')
    for k in cfg.keys():
        if k != 'train_config':
            update_matching_keys(cfg[k], old_cfg[k])
    logger.warning("When resuming, use all sections but 'train_config' are loaded from old config.")


def wandb_init(wandbcfg:config_dict.ConfigDict):    

    if not wandbcfg.name and not FLAGS.wandb_sweep:
        wandbcfg.update({'name':FLAGS.result_folder_name})
    
    cfg_dict = wandbcfg.to_dict()
    input_artifact = cfg_dict.pop('use_artifact')
    logger.debug(f'Arguments passed to wandb.init {cfg_dict}.')

    run = wandb.init(**cfg_dict)

    _pattern = re.compile(".*/FlowReconstructionFromExperiment/local")
    def _ignore_files(path):
        return _pattern.match(path)
    
    run.log_code('.', exclude_fn=_ignore_files)
    
    if input_artifact is not None:
        run.use_artifact(input_artifact)
    
    return run


_keys_to_exclude = [
    'u_train_clean',
    'u_val_clean',
    'train_minmax',
    'val_minmax',
    'u_train',
    'u_val',
    'inn_train',
    'inn_val'
]


def shuffle_batch(rng, n_batch:int, shuffle:bool):
    if not shuffle:
        return rng, jnp.arange(n_batch)
    logger.debug('Shuffling batches each epoch.')
    batch_idx = jax.random.permutation(rng, n_batch)
    [rng] = jax.random.split(rng,1)
    return rng, batch_idx


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
    wandb_run,
    eval_true_mse_train:Callable,
    eval_true_mse_val:Callable,
    yfull_train_batched_clean:Optional[Sequence[jax.Array]],
    yfull_val_clean:Optional[jax.Array],
    shuffle:bool,
):
    '''Train a network'''

    loss_train = []
    loss_val = []
    loss_val_true = []
    loss_div = []
    loss_momentum = []
    loss_sensors = []
    loss_true = []
    loss_val_div = []
    loss_val_momentum = []
    loss_val_sensors = []

    min_loss = np.inf
            
    # compile first
    _ = update(state, rng, x_train_batched[0], y_train_batched[0])
    logger.info('Successfully compiled the update function.')
    error_logged = False

    for i in range(epochs):
        [rng] = jax.random.split(rng,1)

        loss_epoch = []
        loss_epoch_div = []
        loss_epoch_mom = []
        loss_epoch_s = []
        loss_epoch_true = []

        rng, batch_idx = shuffle_batch(rng, n_batch, shuffle)
        
        for b in batch_idx:
            (l, (l_div, l_mom, l_s)), state = update(state, rng, x_train_batched[b], y_train_batched[b])
            if FLAGS.cfg.model_config.dropout_rate > 0.0:
                l, (l_div, l_mom, l_s) = mdl_validation_loss(state.params,None,x_train_batched[b],y_train_batched[b])
                loss_epoch.append(l)
                loss_epoch_div.append(l_div)
                loss_epoch_mom.append(l_mom)
                loss_epoch_s.append(l_s)
            else:
                loss_epoch.append(l)
                loss_epoch_div.append(l_div)
                loss_epoch_mom.append(l_mom)
                loss_epoch_s.append(l_s)

            ## Calculate loss_true = loss_mse_of_all_clean_data+loss_physics
            if i == 0:
                logger.debug('Calculating true loss using clean data.')
            try:
                if not error_logged:
                    l_true = eval_true_mse_train(
                        state.params,
                        None,
                        x_train_batched[b],
                        yfull_train_batched_clean[b]
                    )
            except Exception as e:
                warnings.warn(f'True loss not available.')
                logger.warning(f'True loss cannot be calculated due to {e}')
                l_true = 0.0
                error_logged = True
            loss_epoch_true.append(float(l_true))
            
            if (b == 0 or b == n_batch-1) and i == 0:
                logger.debug(f'batch {b} has size {x_train_batched[b].shape[0]}, loss: {l:.7f}.')

        l_epoch = np.mean(loss_epoch)
        loss_train.append(l_epoch)
        loss_div.append(np.mean(loss_epoch_div))
        loss_momentum.append(np.mean(loss_epoch_mom))
        loss_sensors.append(np.mean(loss_epoch_s))
        loss_true.append(np.mean(loss_epoch_true))

        
        ## Validating
        l_val, l_val_div, l_val_mom, l_val_s, l_val_true = [], [], [], [], []
        for _x,_y,_yclean in zip(x_val,y_val,yfull_val_clean):
            l_v, (l_v_d, l_v_m, l_v_s) = mdl_validation_loss(state.params,None,_x,_y)
            l_val.append(l_v)
            l_val_div.append(l_v_d)
            l_val_mom.append(l_v_m)
            l_val_s.append(l_v_s)

            try:
                if not error_logged:
                    logger.debug('Calculating true validation loss using clean data.')
                    l_v_true = eval_true_mse_val(state.params,None,_x,_yclean)
                else:
                    l_v_true = 0.0
            except Exception as e:
                warnings.warn('True loss not available.')
                logger.warning(f'True loss cannot be calculated due to {e}')
                l_v_true = 0.0
                error_logged = True
            l_val_true.append(l_v_true)

        loss_val.append(np.mean(l_val))
        loss_val_true.append(np.mean(l_val_true))
        loss_val_div.append(np.mean(l_val_div))
        loss_val_momentum.append(np.mean(l_val_mom))
        loss_val_sensors.append(np.mean(l_val_s))


        if wandb_run is not None:
            if i % wandb_run.config.log_frequency == 0:
                logger.debug(f'Logging with wandb every {wandb_run.config.log_frequency} epochs.')
                wandb_run.log({
                    'loss':loss_train[-1],
                    'loss_true':loss_true[-1],
                    'loss_div':loss_div[-1], 
                    'loss_momentum':loss_momentum[-1], 
                    'loss_sensors':loss_sensors[-1],
                    'loss_val':loss_val[-1],
                    'loss_val_true': loss_val_true[-1],
                    'loss_val_div':loss_val_div[-1], 
                    'loss_val_momentum':loss_val_momentum[-1], 
                    'loss_val_sensors':loss_val_sensors[-1],
                    'loss_total': loss_div[-1] + loss_momentum[-1] + loss_sensors[-1],
                    'loss_val_total': loss_val_div[-1] + loss_val_momentum[-1] + loss_val_sensors[-1],
                })

        if l_epoch < min_loss:
            min_loss = l_epoch
            save_trainingstate(tmp_dir,state,'state')

        if i%FLAGS.print == 0:
            print(f'Epoch: {i}, loss: {loss_train[-1]:.7f}, validation_loss: {loss_val[-1]:.7f}', flush=True)
            print(f'    For training, loss_div: {loss_div[-1]:.7f}, loss_momentum: {loss_momentum[-1]:.7f}, loss_sensors: {loss_sensors[-1]:.7f}')
            print(f'    True loss of training: {loss_true[-1]:.7f}, validation: {loss_val_true[-1]:.7f}')


    loss_dict = {
        'loss_train': np.array(loss_train),
        'loss_train_true': np.array(loss_true),
        'loss_div': np.array(loss_div),
        'loss_momentum': np.array(loss_momentum),
        'loss_sensors': np.array(loss_sensors),
        'loss_val': np.array(loss_val),
        'loss_val_true': np.array(loss_val_true),
        'loss_val_div': np.array(loss_val_div),
        'loss_val_momentum': np.array(loss_val_momentum),
        'loss_val_sensors': np.array(loss_val_sensors)
    }

    return state, loss_dict


standard_data_keys = ['u_train_clean', 'u_val_clean', 'train_minmax', 'val_minmax', 'u_train', 'u_val', 'inn_train', 'inn_val', 'y_train', 'y_val']


def main(_):

    cfg = FLAGS.cfg
    wandbcfg = FLAGS.wandbcfg
    if wandbcfg.mode == 'disabled':
        use_wandb = False
    else:
        use_wandb = True
    if FLAGS._experimentcfgstr:
        logger.warning('train.py is started with a pre-set experiment.')
        wandb_updatecfg = get_wandb_config_experiment(FLAGS._experimentcfgstr).config
        wandbcfg.config.update(wandb_updatecfg.to_dict())
    logger.info(f'List of options selected from optional functions: \n      {cfg.case.values()}')

    # ===================== setting up system ==========================
    if FLAGS.gpu_id:
        set_gpu(FLAGS.gpu_id, FLAGS.gpu_mem)
    
    if not FLAGS.result_folder_name:
        _folder = code(cfg.case)
        _folder = _folder + str(time_stamp)
        FLAGS.result_folder_name = _folder

    if FLAGS.chatty:
        logger.setLevel(logging.INFO)
    
    if FLAGS.debug:
        debugger(FLAGS.debug)
        logger.info(f'Running these scripts in debug mode: {FLAGS.debug}.')
    
    if FLAGS.wandb_sweep:
        use_wandb = True
    
    tmp_dir = Path(FLAGS.result_dir,FLAGS.result_folder_name)
    if not tmp_dir.is_dir():
        logger.warning(f'Making a new target directory at {tmp_dir.absolute()}.')
        tmp_dir.mkdir(parents=True)
    else:
        logger.warning(f'Writing into exisiting directory {tmp_dir.absolute()}.')




    if FLAGS.resume:
        if not wandbcfg.resume: # user defined mode first
            wandbcfg.update({'resume': 'must',})
        if cfg.train_config.load_state is None:
            logger.info('Resuming. Loading parameters from current directory because load_state is not specified.')
            load_params_from = Path(tmp_dir) # if resume load from current folder
        else:
            load_params_from = Path(FLAGS.cfg.train_config.load_state) # if not resuming load from previous folder
        update_resume_config(Path(tmp_dir), cfg)
        tmp_dir = Path(tmp_dir,'resume')
        tmp_dir.mkdir(parents=False,exist_ok=False)
        logger.info(f'Loading training state and config from {load_params_from}.')
    elif FLAGS.cfg.train_config.load_state is not None:
        load_params_from = Path(FLAGS.cfg.train_config.load_state) # if not resuming load from previous folder
        logger.info(f'Loading training state from {load_params_from}.')
    else:
        load_params_from = None
        params_old = None
    datacfg = cfg.data_config
    mdlcfg = cfg.model_config
    traincfg = cfg.train_config

    ## Initialise wandb
    if use_wandb:
        logger.info('Updating wandb config with experiment config')
        update_matching_keys(wandbcfg.config, datacfg)
        update_matching_keys(wandbcfg.config, mdlcfg)
        update_matching_keys(wandbcfg.config, traincfg)
        update_matching_keys(wandbcfg.config, cfg.case)
        run = wandb_init(wandbcfg)
        logger.info('Successfully initialised weights and biases.')
        try:
            _datapath = Path(datacfg.data_dir)
            _datatag = "$".join(_datapath.parts)
            _art_name = str(cfg.case._case_dataloader) + ":" + _datatag
            run.use_artifact(_art_name, type="DataPath")
        except Exception as e:
            logger.error(e)
            logger.error("Skip logging DataPath artifact.")

        if FLAGS.wandb_sweep:
            sweep_params = sweep_preprocess_cfg(wandb.config)
            update_matching_keys(datacfg, sweep_params)
            update_matching_keys(mdlcfg, sweep_params)
            update_matching_keys(traincfg, sweep_params)
            FLAGS.result_folder_name = run.name
            logger.info('Running in sweep mode, replace config parameters with sweep parameters.')
            logger.debug(f'Running with {sweep_params}')

    else:
        run = None

    # =================== pre-processing ================================
    
    # data has u_train, u_val, inn_train, inn_val [t, space..., 3] or [t, len]
    logger.info(f'Running case {cfg.case.values()}')
    logger.info('Loading data.')
    data, datainfo = cfg.case.dataloader()
    logger.debug(f'Data dictionary has {[(k, type(k)) if data[k] is not None else (k, None) for k in data.keys()]}')
    logger.debug(f'Datainfo is {datainfo}')

    logger.info('Taking observations.')
    observe_kwargs = {key: value for key, value in data.items() if key not in _keys_to_exclude}
    take_observation, insert_observation = cfg.case.observe(
        datacfg,
        example_pred_snapshot = data['u_train'][0][0,...],
        example_pin_snapshot = data['inn_train'][0][0,...],
        **observe_kwargs
    )
    _, train_minmax = take_observation(np.concatenate(data['u_train'],axis=0), init=True)
    _, val_minmax = take_observation(np.concatenate(data['u_val'],axis=0),init=True)
    observed_train = [take_observation(_u) for _u in data['u_train']]
    observed_val = [take_observation(_u) for _u in data['u_val']]


    data.update({
        'y_train':observed_train,
        'y_val':observed_val,
        'train_minmax':train_minmax,
        'val_minmax':val_minmax 
    })
    logger.debug(f'Data dict now has {data.keys()}')
    data_extra = {k: data[k] for k in data if k not in standard_data_keys}

    percent_observed = 100*(observed_train[0].size/data['u_train'][0].size)
    if run:
        run.config.update({'percent_observed':percent_observed, 're':cfg.data_config.re}, allow_val_change=True)

    # ==================== set up model ==============================
    if traincfg.randseed:
        rng = jax.random.PRNGKey(traincfg.randseed)
        logger.info('Using user assigned random key')
    else:
        rng = jax.random.PRNGKey(int(time_stamp))
        traincfg.update({'randseed':int(time_stamp)})
        cfg.train_config.update({'randseed':int(time_stamp)})


    optimizer = get_optimizer(traincfg)


    prep_data, make_model = cfg.case.select_model(datacfg = datacfg, mdlcfg = mdlcfg, traincfg = traincfg)
    logger.info('Selected a model.')
    
   
    logger.debug(f'The finalised data dictionary has items {data.keys()}')
    data = prep_data(data, datainfo)
    mdl = make_model(mdlcfg)
    logger.info('Made a model.')

    params = mdl.init(rng,data['inn_train'][0][[0],...])
    logger.info('Initialised weights.')
    logger.debug('Params shape')
    logger.debug(jax.tree_util.tree_map(lambda x: x.shape,params))

    # ==================== Optimizer ======================
    opt_state = optimizer.init(params)
    logger.info('Initialised optimiser.')

    # ========= restore model weights and optimizer states if requested =======
    if load_params_from is not None: 
        state_old = restore_trainingstate(load_params_from, 'state')
        # override the new params and opt_state initialised above 
        if Path(load_params_from, 'frozen_params.npy').exists():
            logger.debug('Also loading frozen layers from the same folder.')
            frozen_params = restore_trainingstate(load_params_from, 'frozen_params')
            params_old = params_merge(frozen_params,state_old.params)
        else:
            params_old = state_old.params

    if traincfg.frozen_layers is not None: # Freeze some layers
        params = mdl.load_old_weights(params, params_old)
        logger.debug(f'Model has these layers: {list(params)}')
        logger.info(f'Trying to freeze these layers {traincfg.frozen_layers}')
        params_frozen, params = mdl.freeze_layers(params, list(traincfg.frozen_layers))
        logger.info(f'These layers are frozen: {list(params_frozen)}')
        if len(list(params_frozen)) < 1:
            raise ValueError('No layers have been frozen. Double check the layer names or run in a different training mode.')
        save_trainingstate(tmp_dir, params_frozen, 'frozen_params') # save frozen params
        mdl.set_nontrainable(params_frozen)
        apply_fn = mdl.apply_trainable
        # reinitialize optimizer state to use only trainable weights
        opt_state = optimizer.init(params)
    else: # no layers are frozen, use normal apply and parameters
        if params_old is not None:
            try:
                params = mdl.load_old_weights(params, params_old)
            except AttributeError as e:
                logger.info(f'{e}. Using the old parameters with no modification.')
                params = params_old
        apply_fn = mdl.apply
    
    if FLAGS.resume:
        ## load old opt_state
        logger.info('Getting ready to resume the run by loading the old optimiser state.')
        tree_def_old, leafs_old = zip(*jax.tree_util.tree_leaves_with_path(state_old.opt_state))
        tree_def_old = list(map(lambda x: jax.tree_util.keystr(x), tree_def_old))
        # re.sub("(.*\[')(.*\[.*)", r"\mdl.name/~/\2", tree_key_str)
        def load_optimiser_state(tree_def, leaf_new): # match names
            defstr = jax.tree_util.keystr(tree_def)
            for p, leaf_old in zip(tree_def_old, leafs_old):
                if defstr.split(r'.')[1] == p.split(r'.')[1]:
                    logger.debug(f"Loading optimiser state from {p.split(r'.')[1]}")
                    return leaf_old
            return leaf_new
        opt_state = jax.tree_util.tree_map_with_path(load_optimiser_state, opt_state)


    state = TrainingState(params, opt_state)


    # =================== loss function ==========================

    loss_fn = cfg.case.loss_fn(
        cfg,
        datainfo = datainfo,
        take_observation = take_observation,
        insert_observation = insert_observation,
        **data_extra
    )
    logger.info('Created loss function.')

    mdl_validation_loss = jax.jit(
        jax.tree_util.Partial(
            loss_fn,
            apply_fn,
            apply_kwargs={'training':False},
            y_minmax=data['val_minmax']
        )
    )
    update = generate_update_fn(
        apply_fn,
        optimizer,
        loss_fn,
        kwargs_value_and_grad={'has_aux':True}, 
        kwargs_loss={'y_minmax':data['train_minmax']}
    ) # this update weights once.

    logger.info('MSE of the entire field is used to calculate true loss so we can have an idea of the true performance of the model. It is not used in training.')
    eval_mse_train = jax.jit(
        jax.tree_util.Partial(
            loss_mse,
            apply_fn,
            apply_kwargs={'training':False},
            normalise=datacfg.normalise,
            y_minmax=data['train_minmax']
        )
    )
    eval_mse_val = jax.jit(
        jax.tree_util.Partial(
            loss_mse,
            apply_fn,
            apply_kwargs={'training':False},
            normalise=datacfg.normalise,
            y_minmax=data['val_minmax']
        )
    ) 

    save_config(cfg,tmp_dir)
    # Save command to file
    with open(Path(tmp_dir,"command.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    # ==================== start training ===========================

    logger.debug(f"First batch of input data has shape {data['inn_train'][0].shape}.")
    logger.debug(f"First batch of reference data {data['y_train'][0].shape}.")
    logger.debug(f"First batch of output has shape {apply_fn(state.params, None, data['inn_train'][0], False).shape}")
    _ = update(state,jax.random.PRNGKey(10),data['inn_train'][0],data['y_train'][0])
    logger.info(f"Time taken for each update step {timeit(lambda: update(state, jax.random.PRNGKey(10), data['inn_train'][0], data['y_train'][0]), number=5)}")

    if FLAGS._noisy:
        yfull_batched_clean = data['u_train_clean']
        yfull_val_clean = data['u_val_clean']
    else:
        yfull_batched_clean = data['u_train']
        yfull_val_clean = data['u_val']


    logger.info('Starting training now...')
    state, loss_dict = fit(
        x_train_batched=data['inn_train'],
        y_train_batched=data['y_train'],
        x_val=data['inn_val'],
        y_val=data['y_val'],
        state=state,
        epochs=traincfg.epochs,
        rng=rng,
        n_batch=len(data['inn_train']),
        update=update,
        mdl_validation_loss=mdl_validation_loss,
        tmp_dir=tmp_dir,
        wandb_run=run,
        eval_true_mse_train=eval_mse_train,
        eval_true_mse_val=eval_mse_val,
        yfull_train_batched_clean=yfull_batched_clean,
        yfull_val_clean=yfull_val_clean,
        shuffle=traincfg.shuffle_batch
    )


    logger.info('Finished training.')

    # ===================== Save results
    logger.info(f'writing results to {FLAGS.result_folder_name}')

    with h5py.File(Path(tmp_dir,'results.h5'),'w') as hf:
        for key, value in loss_dict.items():
            hf.create_dataset(key, data=np.array(value))
    

    # ============= Save only best model if doing sweep ==========
    if FLAGS.wandb_sweep:
        api = wandb.Api()

        best_run = api.sweep(f'{run.entity}/{run.project}/{run.sweep_id}').best_run()
        
        try: 
            logger.debug('Trying to save model as an artifact.')
            if np.min(loss_dict['loss_total']) < best_run.summary['loss_total']:
                logger.info('Best model so far, saving weights and configurations.')
                artifact = wandb.Artifact(name=f'sweep_weights_{run.sweep_id}', type='model') 
                artifact.add_dir(tmp_dir)
                run.log_artifact(artifact)
                # run.finish_artifact(artifact) # only necessary for distributed runs
            else:
                logger.info('Not the best model, skip saving weights.')
                artifact = wandb.Artifact(name=f'sweep_weights_{run.sweep_id}', type='model')
                artifact.add_file(Path(tmp_dir,'config.yml'))
                run.log_artifact(artifact)
                for child in tmp_dir.iterdir(): 
                    child.unlink()
                tmp_dir.rmdir()
        except KeyError as e: # probably the first run of the sweep
            logger.warning(e)
            artifact = wandb.Artifact(name=f'sweep_weights_{run.sweep_id}', type='model') 
            artifact.add_dir(tmp_dir)
            run.log_artifact(artifact)
            # run.finish_artifact(artifact)
    
    
    if use_wandb:
        run.finish()
    
    print("Finished at: ", time.asctime(time.localtime(time.time())))





if __name__ == '__main__':

    app.run(main)

