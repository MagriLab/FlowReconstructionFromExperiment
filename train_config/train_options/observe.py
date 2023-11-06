from flowrec._typing import *
from ml_collections.config_dict import ConfigDict

from flowrec.utils.py_helper import slice_from_tuple
from flowrec.data import normalise

import jax
import jax.numpy as jnp


import warnings
import logging
logger = logging.getLogger(f'fr.{__name__}')
import chex


def observe_example(data_config:ConfigDict, **kwargs):
    '''# Example sensor placement function.\n
    
    Pass 'observe@example' to the config to use this observe function in training. Must return two jitable functions, take_obseravtion and insert_observation.\n

    1. take_observation: u -> u_observed.\n
        A function that takes either u_train or u_val, of shape [t,x,y,i] or [t,x,y,z,i], from a dataloader and returns the measurement which will be made available to the network during training.\n
        The returned array must have shape [t, ...], where ... represent any shape.\n

    2. insert_observation: (pred, u_observed) -> pred_with_observed.\n
        A function that takes a prediction from model.apply(inn) and replace the values at sensor locations with u_observed, where u_observed has the shape as returned by function take_observation.\n
        Returns a new array with the same shape as pred.\n
    '''

    def take_observation(u:jax.Array, **kwargs) -> jax.Array:
        raise NotImplementedError 

    def insert_observation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:
        # Available kwargs are:
        # example_pred_snapshot
        # example_pin_snapshot
        raise NotImplementedError

    return take_observation, insert_observation





# ====================== grid ========================

def observe_grid(data_config:ConfigDict, **kwargs):
    '''# Place sensors in a grid.\n
    Sensor locations given by data_config.slice_grid_sensors'''

    s_space = slice_from_tuple(data_config.slice_grid_sensors)
    s = (np.s_[:],) + s_space + (np.s_[:],)

    def take_observation(u:jax.Array,**kwargs) -> jax.Array:
        us = u[s]
        if data_config.normalise:
            logger.info('Normalising observations')
            num_dim = u.shape[-1]
            components = np.squeeze(np.split(us, num_dim, axis=-1))
            components_n, r = normalise(*components)
            us = np.stack(components_n,axis=-1)
        else:
            r = None
        return us, r


    def insert_observation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:
        return pred.at[s].set(observed)
    
    return take_observation, insert_observation


def observe_grid_pin(data_config:ConfigDict,
                    *, 
                    example_pred_snapshot:jax.Array, 
                    example_pin_snapshot:jax.Array,
                    **kwargs):
    '''# Place sensors in a grid and place pressure sensors at the inlet.\n
    Sensor locations given by data_config.slice_grid_sensors, pressure sensor location given by data_config.pressure_inlet_slice.'''

    s_space = slice_from_tuple(data_config.slice_grid_sensors)
    s = (np.s_[:],) + s_space + (np.s_[:],)

    inn_loc, s_pressure = _make_pressure_index(data_config, **kwargs)

    num_sensors = example_pred_snapshot[s_space + (np.s_[:],)].size
    num_pressure = example_pred_snapshot[inn_loc + (np.s_[-1],)].size

    if num_pressure != example_pin_snapshot.size:
        warnings.warn(f'Expect {num_pressure} pressure measurement at inlet, received {example_pin_snapshot.size}. Is this intentional?')
    

    observed_all_shape = (-1,) + example_pred_snapshot[s_space + (np.s_[:],)].shape
    observed_p_shape = (-1,) + example_pred_snapshot[inn_loc+(np.s_[-1],)].shape
    logger.debug(f'Input pressure will be reshaped to {observed_p_shape}')
    

    def take_observation(u:jax.Array,**kwargs) -> jax.Array:
        us = u[s]
        ps = u[s_pressure]

        if data_config.normalise:
            logger.info('Normalising observations')
            num_dim = u.shape[-1]
            components = np.squeeze(np.split(us, num_dim, axis=-1))
            components_u = components[:-1]
            _, r = normalise(*components_u)
            rp = [min(np.min(components[-1]),np.min(ps)), max(np.max(components[-1]),np.max(ps))]
            r.append(np.array(rp))
            
            components_n, _ = normalise(*components, range=r)
            us = np.stack(components_n,axis=-1)

            [ps], _ = normalise(ps, range=[np.array(rp)])
        else:
            r = None
        
        us = us.reshape((-1,num_sensors))
        ps = ps.reshape((-1,num_pressure))
        observed = jnp.concatenate((us,ps), axis=1)
        return observed, r # observed has shape [t,number_of_all_observed]


    def insert_observation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:
        
        us_observed, ps_observed = jnp.array_split(observed,[num_sensors],axis=1)

        us_observed = us_observed.reshape(observed_all_shape)
        ps_observed = ps_observed.reshape(observed_p_shape)

        pred_new = pred.at[s].set(us_observed)
        pred_new = pred_new.at[s_pressure].set(ps_observed)

        return pred_new
    
    return take_observation, insert_observation






# ========================= sparse ======================

def observe_sparse(data_config:ConfigDict, **kwargs):
    '''# Place sensors at specific locations.\n
    Sensor locations are given by data_config.sensor_index.'''

    sensor_idx = data_config.sensor_index

    logger.debug(f'Sensor indices are provided for a {len(sensor_idx)}D flow.')
    if 'example_pred_snapshot' in kwargs.keys():
        chex.assert_rank(kwargs['example_pred_snapshot'][*sensor_idx],2)


    def take_observation(u:jax.Array, **kwargs) -> jax.Array:
        us = u[:,*sensor_idx]
        return us # [snapshot, num_sensors, velocities]

    def insert_observation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:
        pred_new = pred.at[:,*sensor_idx].set(observed)
        return pred_new

    return take_observation, insert_observation


def observe_sparse_pin(data_config:ConfigDict,
                        *, 
                        example_pred_snapshot:jax.Array, 
                        example_pin_snapshot:jax.Array,
                        **kwargs):
    '''# Place sensors at specific locations.\n
    Sensor locations given by data_config.sensor_index, pressure sensor location given by data_config.pressure_inlet_slice.'''

    # velocity and pressure sensors
    sensor_idx = data_config.sensor_index
    # num_sensors = len(sensor_idx[0])
    num_sensors = example_pred_snapshot[*sensor_idx].size
    logger.debug(f'Sensor indices are provided for a {len(sensor_idx)}D flow, with {num_sensors} measurements.')
    observed_all_shape = (-1,) + example_pred_snapshot[*sensor_idx].shape
    logger.debug(f'Observed will have shape {observed_all_shape}.')
    
    chex.assert_rank(example_pred_snapshot[*sensor_idx],2)
    
    inn_loc, s_pressure = _make_pressure_index(data_config, **kwargs)

    num_pressure = example_pred_snapshot[inn_loc + (np.s_[-1],)].size
    logger.debug(f'Number of pressure sensors at the inlet is {num_pressure}.')
    observed_p_shape = (-1,) + example_pred_snapshot[inn_loc + (np.s_[-1],)].shape
    
    if num_pressure != example_pin_snapshot.size:
        warnings.warn(f'Expect {num_pressure} pressure measurement at inlet, received {example_pin_snapshot.size}. Is this intentional?')
    

    def take_observation(u:jax.Array, **kwargs) -> jax.Array:
        us = u[:,*sensor_idx].reshape((-1,num_sensors))
        ps = u[s_pressure].reshape((-1,num_pressure))
        observed = jnp.concatenate((us,ps), axis=1)
        logger.debug(f'The observed has shape {observed.shape}')
        return observed # observed has shape [t,number_of_all_observed]

    def insert_observation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:
        us_observed, ps_observed = jnp.array_split(observed,[num_sensors],axis=1)

        us_observed = us_observed.reshape(observed_all_shape)
        ps_observed = ps_observed.reshape(observed_p_shape)

        pred_new = pred.at[s_pressure].set(ps_observed)
        pred_new = pred_new.at[:,*sensor_idx].set(us_observed)

        return pred_new

    return take_observation, insert_observation



def _make_pressure_index(data_config, **kwargs):

    if data_config.pressure_inlet_slice:
        inn_loc = slice_from_tuple(data_config.pressure_inlet_slice)
        s_pressure = (np.s_[:],) + inn_loc + (np.s_[-1],)
        logger.debug("Using 'pressure_inlet_slice' to locate pressure sensors.")
    elif '_slice_inn' in kwargs:
        s_pressure = kwargs['_slice_inn']
        inn_loc = s_pressure[1:-1]
        logger.debug("Using '_slice_inn' founc in data to locate pressure sensors.")

    return inn_loc, s_pressure

