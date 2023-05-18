from flowrec._typing import *
from ml_collections.config_dict import ConfigDict

from utils import simulation2d
from utils.py_helper import slice_from_tuple
from flowrec.data import data_partition, unnormalise_group, normalise,get_whitenoise_std

import jax.numpy as jnp
from jax.tree_util import Partial

from typing import Callable, Sequence

import warnings
import logging
logger = logging.getLogger(f'fr.{__name__}')
import chex

from absl import flags
FLAGS = flags.FLAGS


def dataloader_example(data_config:ConfigDict):

    '''# Example dataloader function.\n
    
    Load data and returns a dictionary with the following arrays. Pass 'dataloader@example' to the config to use this dataloader in training. \n

    The function must return a dictionary of datasets and a datainfo. \n

    The dictionary must have the following items.\n
        1. u_train: [t,x,y,i] or [t,x,y,z,i]. i is the number of variables. The last dimension must be ordered u,v,p or u,v,w,p.\n
        2. u_val: same rules as u_train.\n
        3. inn_train: input vector [t,j], j is the number of inputs and t is the number of snapshots.\n
        4. inn_val: same rules as inn_train.\n 
        
        If normalising the components, then the dictionary needs to also contain\n
            1. train_minmax is the shape of [(min, max of u), (min, max of v), ..., (min, max of pressure)]. \n
            2. val_minmax with the same shape as train_minmax. \n

    The second returned variable, datainfo, should be a MetadataTree.\n
    '''
    pass




def dataloader_2dtriangle(cfg:ConfigDict) -> dict:
    '''# Load data base on data_config. 
    For use with the generated 2D wake behind the triangle, any Reynolds number.\n
    
    data:\n
        - u_train: pre-processed generated data, has shape [t,x,y,3].\n
        - u_val: same shape as u_train.\n
        - inn_train: pre-processed pressure measurement at the base of the triangle, has shape [t,len].\n
        - inn_val: same shape as inn_train.\n
        - If normalise, data dict also has train_minmax and val_minmax [3,range], range is [min,max]\n
        - If noisy, data dict also has u_train_clean and u_val_clean.
    
    datainfo: of class _Metadata2d.
    
    '''
    data = {}

    x_base = 132
    triangle_base_coords = [49,80]
    logger.debug(f'The base of the triangle is placed at x={x_base}, between y={triangle_base_coords[0]} and {triangle_base_coords[1]}.')

    (ux,uy,pp) = simulation2d.read_data(cfg.data_dir,x_base)
    x = np.stack([ux,uy,pp],axis=0)
    logger.debug(f'Simulated wake has shape {x.shape}.')
    # remove parts where uz is not zero
    s = slice_from_tuple(cfg.slice_to_keep)
    x = x[s]
    logger.debug(f'Out of the entire simulated wake, the useful domain has shape {x.shape}, determined by data_config.slice_to_keep.')

    if cfg.shuffle:
        if not cfg.randseed:
            randseed = np.random.randint(1,10000)
            cfg.update({'randseed':randseed})
            logger.info('Shuffling data with a newly generated random key.')
        else:
            randseed = cfg.randseed
            logger.info('Shuffling data with the randon key provided in data_config.')
    else:
        randseed = None

    # Add white noise
    if cfg.snr:
        logger.info('Saving clean data for calculating true loss')
        [x_train,x_val,_], _ = data_partition(x,1,cfg.train_test_split,REMOVE_MEAN=cfg.remove_mean,randseed=randseed,SHUFFLE=cfg.shuffle) # Do not shuffle, do not remove mean for training with physics informed loss
        [ux_train,uy_train,pp_train] = np.squeeze(np.split(x_train,3,axis=0))
        [ux_val,uy_val,pp_val] = np.squeeze(np.split(x_val,3,axis=0))
        u_train = np.stack((ux_train,uy_train,pp_train),axis=-1)
        u_val = np.stack((ux_val,uy_val,pp_val),axis=-1)
        data.update({
            'u_train_clean': u_train,
            'u_val_clean': u_val
        })
        

        logger.info('Adding white noise to data.')
        FLAGS._noisy = True
        std_data = np.std(x,axis=(1,2,3),ddof=1)
        # snr_l = 10.**(cfg.snr/10.)
        # std_n = np.sqrt(std_data**2/snr_l)
        std_n = get_whitenoise_std(cfg.snr,std_data)
        noise_ux = np.random.normal(scale=std_n[0],size=x[0,...].shape)
        noise_uy = np.random.normal(scale=std_n[1],size=x[1,...].shape)
        noise_pp = np.random.normal(scale=std_n[2],size=x[2,...].shape)
        noise = np.stack([noise_ux,noise_uy,noise_pp],axis=0)
        x = x + noise

    
    [x_train,x_val,_], _ = data_partition(x,1,cfg.train_test_split,REMOVE_MEAN=cfg.remove_mean,randseed=randseed,SHUFFLE=cfg.shuffle) # Do not shuffle, do not remove mean for training with physics informed loss
    logger.info(f'Remove mean is {cfg.remove_mean}')

    [ux_train,uy_train,pp_train] = np.squeeze(np.split(x_train,3,axis=0))
    [ux_val,uy_val,pp_val] = np.squeeze(np.split(x_val,3,axis=0))
    
    # Normalise
    if cfg.normalise:
        logger.info('Normalising input.')
        [ux_train,uy_train,pp_train], train_minmax = normalise(ux_train,uy_train,pp_train)
        [ux_val,uy_val,pp_val], val_minmax = normalise(ux_val,uy_val,pp_val)
    else:
        train_minmax = []
        val_minmax = []
    
    data.update({
        'train_minmax': jnp.asarray(train_minmax),
        'val_minmax': jnp.asarray(val_minmax),
    })

    
    pb_train = simulation2d.take_measurement_base(pp_train,ly=triangle_base_coords,centrex=0)
    pb_val = simulation2d.take_measurement_base(pp_val,ly=triangle_base_coords,centrex=0)

    # information about the grid
    datainfo = DataMetadata(
        re = cfg.re,
        discretisation=[cfg.dt,cfg.dx,cfg.dy],
        axis_index=[0,1,2],
        problem_2d=True
    ).to_named_tuple()
    logger.debug(f'datainfo is {datainfo}.')

    pb_train = np.reshape(pb_train,(cfg.train_test_split[0],-1))
    pb_val = np.reshape(pb_val,(cfg.train_test_split[1],-1))
    logger.debug(f'Input of the training set has shape {pb_train.shape}.')

    u_train = np.stack((ux_train,uy_train,pp_train),axis=-1)
    u_val = np.stack((ux_val,uy_val,pp_val),axis=-1)
    logger.debug(f'Shapes of u_train and u_val are {u_train.shape} and {u_val.shape}.')


    data.update({
        'u_train': u_train, # [t,x,y,3]
        'u_val': u_val, # [t,x,y,3]
        'inn_train': pb_train, # [t,len]
        'inn_val': pb_val # [t,len]
    })

    return data, datainfo

