from flowrec._typing import *
from ml_collections.config_dict import ConfigDict

from flowrec.utils.py_helper import slice_from_tuple
from flowrec.data import normalise
from flowrec.sensors import random_coords_generator

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

        if ('init' in kwargs) and (kwargs['init'] is True):
            # return observed, range
            raise NotImplementedError 
        
        # return observed
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

        if ('init' in kwargs) and (kwargs['init'] is True):
            logger.debug(f'{us[0,...].size} observations for each snapshot, has shape {us[0,...].shape}')
            if data_config.normalise:
                # logger.info('Normalising observations')
                num_dim = u.shape[-1]
                components = np.squeeze(np.split(us, num_dim, axis=-1))
                _, r = normalise(*components)
                # us = np.stack(components_n,axis=-1)
            else:
                r = None
            return us, r

        return us

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

        if ('init' in kwargs) and (kwargs['init'] is True):
            if data_config.normalise:
                # logger.info('Normalising observations')
                num_dim = u.shape[-1]
                components = np.squeeze(np.split(us, num_dim, axis=-1))
                components_u = components[:-1]
                _, r = normalise(*components_u)
                rp = [min(np.min(components[-1]),np.min(ps)), max(np.max(components[-1]),np.max(ps))]
                r.append(np.array(rp))
                
                # components_n, _ = normalise(*components, range=r)
                # us = np.stack(components_n,axis=-1)

                # [ps], _ = normalise(ps, range=[np.array(rp)])
            else:
                r = None
            
            us = us.reshape((-1,num_sensors))
            ps = ps.reshape((-1,num_pressure))
            observed = jnp.concatenate((us,ps), axis=1)
            return observed, r # observed has shape [t,number_of_all_observed]
        
        us = us.reshape((-1,num_sensors))
        ps = ps.reshape((-1,num_pressure))
        observed = jnp.concatenate((us,ps), axis=1)
        return observed # observed has shape [t,number_of_all_observed]


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

        if ('init' in kwargs) and (kwargs['init'] is True):
            if data_config.normalise:
                # logger.info('Normalising observations')
                num_dim = u.shape[-1]
                components = np.squeeze(np.split(us, num_dim, axis=-1))
                _, r = normalise(*components)
                # us = np.stack(components_n,axis=-1)
            else:
                r = None
            return us, r 
        
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
        us = u[:,*sensor_idx]
        ps = u[s_pressure]

        if ('init' in kwargs) and (kwargs['init'] is True):

            if data_config.normalise:
                # logger.info('Normalising observations')
                num_dim = u.shape[-1]
                components = np.squeeze(np.split(us, num_dim, axis=-1))
                components_u = components[:-1]
                _, r = normalise(*components_u)
                rp = [min(np.min(components[-1]),np.min(ps)), max(np.max(components[-1]),np.max(ps))]
                r.append(np.array(rp))
                # components_n, _ = normalise(*components, range=r)
                # us = np.stack(components_n,axis=-1)
                # [ps], _ = normalise(ps, range=[np.array(rp)])
            else:
                r = None

            us = us.reshape((-1,num_sensors))
            ps = ps.reshape((-1,num_pressure))
            observed = jnp.concatenate((us,ps), axis=1)
            logger.debug(f'The observed has shape {observed.shape}')

            return observed, r # observed has shape [t,number_of_all_observed]

        us = us.reshape((-1,num_sensors))
        ps = ps.reshape((-1,num_pressure))
        observed = jnp.concatenate((us,ps), axis=1)
        return observed

    
    def insert_observation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:
        us_observed, ps_observed = jnp.array_split(observed,[num_sensors],axis=1)

        us_observed = us_observed.reshape(observed_all_shape)
        ps_observed = ps_observed.reshape(observed_p_shape)

        pred_new = pred.at[s_pressure].set(ps_observed)
        pred_new = pred_new.at[:,*sensor_idx].set(us_observed)

        return pred_new

    return take_observation, insert_observation



# ====================== random sparse ================================
def observe_random_pin(
        data_config:ConfigDict,
        *, 
        example_pred_snapshot:jax.Array, 
        example_pin_snapshot:jax.Array,
        **kwargs
):
    
    # dimensions of the problem
    gridsize_space = example_pred_snapshot.shape[:-1]
    dim = len(gridsize_space)
    if (data_config.dz is not None) and (dim != 3):
        logger.warning(f'Expect 3D data, got grid size {gridsize_space}.')
    elif (data_config.dz is None) and (dim !=2 ):
        logger.warning(f'Expect 2D data, got grid size {gridsize_space}.')
        
    sensor_seed, num_sensors = data_config.random_sensors
    logger.info(f'Creating {num_sensors} random sensors on a grid of {gridsize_space}')
    rng = np.random.default_rng(sensor_seed)
    sensor_idx_np = random_coords_generator(
        rng,
        num_sensors,
        gridsize_space
    )
    sensor_idx = tuple(tuple(_arr) for _arr in [*sensor_idx_np])

    data_config.update({'sensor_index': sensor_idx})

    logger.debug("Updated data_config.sensor_index with random index and calling 'oberve_sparse_pin'.")
    return observe_sparse_pin(data_config, example_pred_snapshot=example_pred_snapshot, example_pin_snapshot=example_pin_snapshot, **kwargs)


# ===================== slice of velocities from a 3D domain ===============

def observe_slice(
        data_config:ConfigDict,
        *,
        example_pred_snapshot:jax.Array, 
        **kwargs
):
    # keep dimensions
    grid_shape = example_pred_snapshot.shape[:-1]
    dim = len(grid_shape)
    if (data_config.dz is not None) and (dim != 3):
        logger.warning(f'Expect 3D data, got grid size {grid_shape}.')
    
    x, y, z, num_components = data_config.measure_slice

    _slice_index = np.s_[:,x,y,z,:num_components]
    s = tuple([slice(None,None,None) if a is None else a for a in _slice_index])
    slice_shape = example_pred_snapshot[s[1:]].shape
    logger.debug(f'The slice has shape {slice_shape}, index {s}')
    num_sensors = np.prod(slice_shape)
    
    def take_observation(u:jax.Array, **kwargs) -> jax.Array:
        us = u[s]
        # ps = u[s_pressure]


        if ('init' in kwargs) and (kwargs['init'] is True):
            if data_config.normalise:
                # logger.info('Normalising observations')
                num_dim = u.shape[-1]
                components = np.squeeze(np.split(us, num_dim, axis=-1))
                if num_dim == dim+1:
                    # pressure and velocity are both on the slice
                    # components_u = components[:-1]
                    _, r = normalise(*components)
                    # rp = [min(np.min(components[-1]),np.min(ps)), max(np.max(components[-1]),np.max(ps))]
                    # r.append(np.array(rp))
                elif num_dim == dim:
                    # only velocities (and all velocities compoenets) on the slice
                    _, r = normalise(*components)
                    # r.append([ps.min(), ps.max()])
                else:
                    # one velocity is missing
                    logger.error('Nomralisation is not defined when one or more velocity components are not measured.')
                    r = None
            else:
                r = None

            return us, r # observed has shape [t,number_of_all_observed]
        
        return us # observed has shape [t,number_of_all_observed]

    def insert_obervation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:


        pred_new = pred.at[s].set(observed)
        return pred_new
    
    return take_observation, insert_obervation
 





def observe_slice_pin(
        data_config:ConfigDict,
        *,
        example_pred_snapshot:jax.Array, 
        example_pin_snapshot:jax.Array,
        **kwargs
):
    # keep dimensions
    grid_shape = example_pred_snapshot.shape[:-1]
    dim = len(grid_shape)
    if (data_config.dz is not None) and (dim != 3):
        logger.warning(f'Expect 3D data, got grid size {grid_shape}.')
    
    x, y, z, num_components = data_config.measure_slice

    _slice_index = np.s_[:,x,y,z,:num_components]
    s = tuple([slice(None,None,None) if a is None else a for a in _slice_index])
    slice_shape = example_pred_snapshot[s[1:]].shape
    logger.debug(f'The slice has shape {slice_shape}, index {s}')
    num_sensors = np.prod(slice_shape)
    inn_loc, s_pressure = _make_pressure_index(data_config, **kwargs)
    pressure_shape = example_pred_snapshot[inn_loc + (-1,)].shape
    num_pressure = np.prod(pressure_shape)
    if num_pressure != example_pin_snapshot.size:
        warnings.warn(f'Expect {num_pressure} pressure measurement at inlet, received {example_pin_snapshot.size}. Is this intentional?')
    
    def take_observation(u:jax.Array, **kwargs) -> jax.Array:
        us = u[s]
        ps = u[s_pressure]


        if ('init' in kwargs) and (kwargs['init'] is True):
            if data_config.normalise:
                # logger.info('Normalising observations')
                num_dim = u.shape[-1]
                components = np.squeeze(np.split(us, num_dim, axis=-1))
                if num_dim == dim+1:
                    # pressure and velocity are both on the slice
                    components_u = components[:-1]
                    _, r = normalise(*components_u)
                    rp = [min(np.min(components[-1]),np.min(ps)), max(np.max(components[-1]),np.max(ps))]
                    r.append(np.array(rp))
                elif num_dim == dim:
                    # only velocities (and all velocities compoenets) on the slice
                    _, r = normalise(*components)
                    r.append([ps.min(), ps.max()])
                else:
                    # one velocity is missing
                    logger.error('Nomralisation is not defined when one or more velocity components are not measured.')
                    r = None
            else:
                r = None
            
            us = us.reshape((-1,num_sensors))
            ps = ps.reshape((-1,num_pressure))
            observed = jnp.concatenate((us,ps), axis=1)
            return observed, r # observed has shape [t,number_of_all_observed]
        
        us = us.reshape((-1,num_sensors))
        ps = ps.reshape((-1,num_pressure))
        observed = jnp.concatenate((us,ps), axis=1)
        return observed # observed has shape [t,number_of_all_observed]

    def insert_obervation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:

        us_observed, ps_observed = jnp.array_split(observed,[num_sensors],axis=1)

        us_observed = us_observed.reshape((-1,)+slice_shape)
        ps_observed = ps_observed.reshape((-1,)+pressure_shape)

        pred_new = pred.at[s].set(us_observed)
        pred_new = pred_new.at[s_pressure].set(ps_observed)
        return pred_new
    
    return take_observation, insert_obervation
                    


def observe_boxtest(
        data_config:ConfigDict,
        *,
        example_pred_snapshot:jax.Array, 
        example_pin_snapshot:jax.Array,
        **kwargs
):
    # keep dimensions
    grid_shape = example_pred_snapshot.shape[:-1]
    dim = len(grid_shape)
    if dim != 3:
        logger.error(f'Expect 3D data.')

    plane1 = data_config.plane1[:-1]
    n1 = list(data_config.plane1[-1])
    plane2 = data_config.plane2[:-1]
    n2 = list(data_config.plane2[-1])
    s1 = tuple([slice(None,None,None) if a is None else a for a in np.s_[*plane1]])
    s2 = tuple([slice(None,None,None) if a is None else a for a in np.s_[*plane2]])
    
    # take planes
    _empty_data = np.zeros_like(example_pred_snapshot,dtype=int)
    # _empty_data[*s1,n1] = 1
    # _empty_data[*s2,n2] = 1
    _empty_data[:20,20:41,20:41,:3] = 1
    _empty_data[1:19,21:40,21:40,:3] = 0

    x, y, z, u = np.indices(_empty_data.shape)
    has_values = _empty_data > 0
    x1 = x[has_values]
    y1 = y[has_values]
    z1 = z[has_values]
    u1 = u[has_values]
    idx = np.array(tuple(zip(x1,y1,z1,u1))).T #(4,num)
    s = np.s_[:,*idx]
    
    slice_shape = example_pred_snapshot[s[1:]].shape
    num_sensors = idx.shape[1]
    logger.debug(f'The slice has shape {slice_shape}, {num_sensors} sensors in total.')


    # inlet
    inn_loc, s_pressure = _make_pressure_index(data_config, **kwargs)
    pressure_shape = example_pred_snapshot[inn_loc + (-1,)].shape
    num_pressure = np.prod(pressure_shape)
    if num_pressure != example_pin_snapshot.size:
        warnings.warn(f'Expect {num_pressure} pressure measurement at inlet, received {example_pin_snapshot.size}. Is this intentional?')

    def take_observation(u:jax.Array, **kwargs) -> jax.Array:
        us = u[s]
        ps = u[s_pressure]
        us = us.reshape((-1,num_sensors))
        ps = ps.reshape((-1,num_pressure))
        observed = jnp.concatenate((us,ps), axis=1)

        if ('init' in kwargs) and (kwargs['init'] is True):
            if data_config.normalise:
                _u1 = us[:,has_values[...,0]]
                _u2 = us[:,has_values[...,1]]
                _u3 = us[:,has_values[...,2]]
                _p = us[:,has_values[...,3]].flatten()
                _p = np.concatenate((_p,ps.faltten()))
                r = []
                r.append([_u1.min(), _u1.max()])
                r.append([_u2.min(), _u2.max()])
                r.append([_u3.min(), _u3.max()])
                r.append([_p.min(), _p.max()])
            else:
                r = None
            return observed, r

        return observed # observed has shape [t,number_of_all_observed]
    

    def insert_obervation(pred:jax.Array, observed:jax.Array, **kwargs) -> jax.Array:

        us_observed, ps_observed = jnp.array_split(observed,[num_sensors],axis=1)

        us_observed = us_observed.reshape((-1,)+slice_shape)
        ps_observed = ps_observed.reshape((-1,)+pressure_shape)

        pred_new = pred.at[s].set(us_observed)
        pred_new = pred_new.at[s_pressure].set(ps_observed)
        return pred_new
    
    return take_observation, insert_obervation


# =====================================================================

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

