from flowrec._typing import *
from ml_collections.config_dict import ConfigDict

from utils import simulation2d
from utils.py_helper import slice_from_tuple
from flowrec.data import data_partition
from flowrec import losses
from flowrec.models import cnn, feedforward
from flowrec import physics_and_derivatives as derivatives

import jax
import jax.numpy as jnp

from typing import Callable, Sequence
from haiku import Params


def _dataloader_opt(data_config:ConfigDict) -> dict:
    '''Example dataloader function.\n
    
    Load data and returns a dictionary with the following arrays. Pass 'dataloader@opt' to the config to use this dataloader in training. \n

    u_train: [t,x,y,i] or [t,x,y,z,i]. i is the number of variables. The last dimension must be ordered u,v,p or u,v,w,p.\n
    u_val: same rules as u_train.\n
    inn_train: input vector [t,j], j is the number of inputs and t is the number of snapshots.\n
    inn_val: same rules as inn_train
    '''
    pass







# ====================== Dataloader ==========================

def dataloader_2dtriangle(cfg:ConfigDict) -> dict:
    '''Load data base on data_config.'''
    # u_train (tensor), u_val (tensor), inn_train (vector), inn_val (vector)
    # datainfo
    x_base = 132
    triangle_base_coords = [49,80]

    (ux,uy,pp) = simulation2d.read_data(cfg.data_dir,x_base)
    x = np.stack([ux,uy,pp],axis=0)
    # remove parts where uz is not zero
    s = slice_from_tuple(cfg.slice_to_keep)
    x = x[s]

    if cfg.shuffle:
        randseed = np.random.randint(1,10000)
        cfg.update({'randseed':randseed})
    else:
        randseed = None

    
    [x_train,x_val,_], _ = data_partition(x,1,cfg.train_test_split,REMOVE_MEAN=cfg.remove_mean,randseed=randseed,SHUFFLE=cfg.shuffle) # Do not shuffle, do not remove mean for training with physics informed loss

    [ux_train,uy_train,pp_train] = np.squeeze(np.split(x_train,3,axis=0))
    [ux_val,uy_val,pp_val] = np.squeeze(np.split(x_val,3,axis=0))
    
    pb_train = simulation2d.take_measurement_base(pp_train,ly=triangle_base_coords,centrex=0)
    pb_val = simulation2d.take_measurement_base(pp_val,ly=triangle_base_coords,centrex=0)

    # information about the grid
    datainfo = DataMetadata(
        re = cfg.re,
        discretisation=[cfg.dt,cfg.dx,cfg.dy],
        axis_index=[0,1,2],
        problem_2d=True
    ).to_named_tuple()

    pb_train = np.reshape(pb_train,(cfg.train_test_split[0],-1))
    pb_val = np.reshape(pb_val,(cfg.train_test_split[1],-1))

    u_train = np.stack((ux_train,uy_train,pp_train),axis=-1)
    u_val = np.stack((ux_val,uy_val,pp_val),axis=-1)


    data = {
        'u_train': u_train, # [t,x,y,3]
        'u_val': u_val, # [t,x,y,3]
        'inn_train': pb_train, # [t,len]
        'inn_val': pb_val # [t,len]
    }

    return data, datainfo


# ======================= Sensor placement ========================

def observe_grid(data_config:ConfigDict):
    s_space = slice_from_tuple(data_config.slice_grid_sensors)
    s = (np.s_[:],) + s_space + (np.s_[:],)

    def take_observation(u:jax.Array,**kwargs) -> jax.Array:
        return u[s]


    def insert_observation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:
        return pred.at[s].set(observed)
    
    return take_observation, insert_observation


# ======================= Model =============================

def select_model_ffcnn(**kwargs):

    def prep_data(data:dict) -> dict:
        # make data into suitable form
        # data.update({'u_train':new_u_train,'inn_train':new_inn_train})
        return data


    def make_model(cfg:ConfigDict) -> BaseModel:
        mdl = cnn.Model(
            mlp_layers = cfg.mlp_layers,
            output_shape = cfg.output_shape,
            cnn_channels = cfg.cnn_channels,
            cnn_filters = cfg.cnn_filters,
            dropout_rate = cfg.dropout_rate
        )
        return mdl

    return prep_data, make_model




# ======================= Loss Function ===================

def loss_fn_physicswithdata(cfg,**kwargs):
    '''Insert observations into prediction, then train on physics loss (or physics+sensor loss).'''

    take_observation: Callable = kwargs['take_observation']
    insert_observation: Callable = kwargs['insert_observation']
    
    datainfo = kwargs['datainfo']

    wp = cfg.train_config.weight_physics
    ws = cfg.train_config.weight_sensors

    def loss_fn(apply_fn:Callable ,params:Params, rng:jax.random.PRNGKey, x:Sequence[jax.Array], y:Sequence[jax.Array],**kwargs):
        pred = apply_fn(params, rng, x, **kwargs)
        pred_observed = take_observation(pred)
        loss_sensor = losses.mse(pred_observed, y)

        pred_new = insert_observation(pred,y)
        loss_div = losses.divergence(pred_new[...,0], pred_new[...,1], datainfo)
        mom_field = derivatives.momentum_residue_field(
                            ux=pred_new[...,0],
                            uy=pred_new[...,1],
                            p=pred_new[...,2],
                            datainfo=datainfo) # [i,t,x,y]
        loss_mom = jnp.mean(mom_field**2)*mom_field.shape[0]
        
        
        return wp*(loss_div+loss_mom)+ws*loss_sensor, (loss_div,loss_mom,loss_sensor)
    
    return loss_fn


def loss_fn_physicsnoreplace(cfg,**kwargs):
    '''Train on physics loss + sensor loss.'''

    take_observation:Callable = kwargs['take_observation']
    datainfo = kwargs['datainfo']

    wp = cfg.train_config.weight_physics
    ws = cfg.train_config.weight_sensors

    def loss_fn(apply_fn:Callable ,params:Params, rng:jax.random.PRNGKey, x:Sequence[jax.Array], y:Sequence[jax.Array],**kwargs):
        pred = apply_fn(params, rng, x, **kwargs)
        pred_observed = take_observation(pred)
        loss_sensor = losses.mse(pred_observed, y)

        loss_div = losses.divergence(pred[...,0], pred[...,1], datainfo)
        mom_field = derivatives.momentum_residue_field(
                            ux=pred[...,0],
                            uy=pred[...,1],
                            p=pred[...,2],
                            datainfo=datainfo) # [i,t,x,y]
        loss_mom = jnp.mean(mom_field**2)*mom_field.shape[0]
        
        return wp*(loss_div+loss_mom)+ws*loss_sensor, (loss_div,loss_mom,loss_sensor)
    
    return loss_fn    


def _dummy():
    print('a fake function.')
    return 1