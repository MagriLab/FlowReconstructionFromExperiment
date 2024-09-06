"""Functions for post processing the results"""
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RBFInterpolator
from typing import Callable, List, Tuple
from ml_collections import ConfigDict
from .sensors import griddata_periodic
from ._typing import Array

class Interpolator():
    def __init__(self) -> None:
        """Interpolate the flow fields from sparse measurements"""
        pass

    @staticmethod
    def triangle2d(u: Array, pb: Array, case_observe:Callable, datacfg:ConfigDict, kernel:str = 'thin_plate_spline'):
        """Interpolate a 2dtraingle dataset from random sensors.
        
        --------------------------
        u: the training dataset, clean or noisy \n
        pb: inlet pressure taken from the same dataset \n
        case_observe: config.case.observe function \n
        datacfg: config.data_config \n
        """
        take_observation, insert_observation = case_observe(datacfg, example_pred_snapshot=u[0,...],example_pin_snapshot=pb[0,...])
        observed = take_observation(u)
        temp_observed = np.empty_like(u)
        temp_observed.fill(np.nan) #this is noisy
        temp_observed = insert_observation(jnp.asarray(temp_observed),jnp.asarray(observed)) # observed_test is noisy if

        # get sensor coordinates
        sensors_empty = np.empty_like(u[[0],...])
        sensors_empty.fill(np.nan)

        grid_x,grid_y = np.mgrid[0:u[...,0].shape[1], 0:u[...,0].shape[2]]

        gridx1 = np.repeat(grid_x[None,:,:,None],3,axis=3)
        gridy1 = np.repeat(grid_y[None,:,:,None],3,axis=3)

        idx_x = take_observation(gridx1)
        idx_y = take_observation(gridy1)

        idx_x = insert_observation(jnp.asarray(sensors_empty),jnp.asarray(idx_x))[0,...]
        sensors_loc_x = []
        for i in range(idx_x.shape[-1]):
            sensors_loc_x.append(idx_x[...,i][~np.isnan(idx_x[...,i])])

        idx_y = insert_observation(jnp.asarray(sensors_empty),jnp.asarray(idx_y))[0,...]
        sensors_loc_y = []
        for i in range(idx_y.shape[-1]):
            sensors_loc_y.append(idx_y[...,i][~np.isnan(idx_y[...,i])])


        compare_interp = list([])
        nt = u.shape[0]
        _locs = np.stack((grid_x.flatten(),grid_y.flatten()),axis=-1)

        print('Starting interpolation')
        for i in range(3):
            sensors_loc = np.stack((sensors_loc_x[i].flatten(),sensors_loc_y[i].flatten()),axis=-1)
            for j in range(nt):
                temp_measurement = temp_observed[j,...,i][~np.isnan(temp_observed[j,...,i])]
                rbf = RBFInterpolator(sensors_loc,temp_measurement.flatten(),kernel=kernel)
                _interp = rbf(_locs).reshape(grid_x.shape)
                compare_interp.append(_interp)
        compare_interp = np.array(compare_interp)
        compare_interp = np.stack((compare_interp[:nt,...],compare_interp[nt:2*nt,...],compare_interp[2*nt:3*nt,...]),axis=-1)

        return compare_interp, temp_observed
    

    @staticmethod
    def kolmogorov2d(insert_observation_fn:Callable, sensor_locs:List, data_shape:Tuple, observed:Array, kernel:str = 'cubic'):

        sensors_loc_x, sensors_loc_y = sensor_locs

        compare_interp = []
        nt = data_shape[0]
        ndim = data_shape[-1]

        side_length = data_shape[1]
        g1,g2 = np.mgrid[-side_length:side_length*2, -side_length:side_length*2]
        
        temp_observed = np.empty(data_shape)
        temp_observed.fill(np.nan) #this is noisy
        temp_observed = insert_observation_fn(jnp.asarray(temp_observed),jnp.asarray(observed)) # observed_test is noisy if

        for i in range(ndim):
            _locs = np.stack((sensors_loc_x[i].flatten(),sensors_loc_y[i].flatten()),axis=1)
            for t in range(nt):
                _interp = griddata_periodic(_locs,temp_observed[t,...,i][~np.isnan(temp_observed[t,...,i])],(g1,g2),kernel,side_length)
                compare_interp.append(_interp[side_length:2*side_length,side_length:2*side_length])

        compare_interp = np.array(compare_interp)
        if ndim > 1:
            compare_interp = np.stack((compare_interp[:nt,...],compare_interp[nt:2*nt,...],compare_interp[2*nt:3*nt,...]),axis=-1)

        return compare_interp, temp_observed


def get_sensor_locs_2d(example_train:Array, take_observation_fn:Callable, insert_observation_fn:Callable):

    sensors_empty = np.empty_like(example_train[[0],...])
    sensors_empty.fill(np.nan)
    grid_x, grid_y = np.mgrid[0:example_train.shape[1], 0:example_train.shape[2]]

    gridx1 = np.repeat(grid_x[None,:,:,None],3,axis=3)
    gridy1 = np.repeat(grid_y[None,:,:,None],3,axis=3)

    idx_x = take_observation_fn(gridx1)
    idx_y = take_observation_fn(gridy1)

    idx_x = insert_observation_fn(jnp.asarray(sensors_empty),jnp.asarray(idx_x))[0,...]
    sensors_loc_x = []
    for i in range(idx_x.shape[-1]):
        sensors_loc_x.append(idx_x[...,i][~np.isnan(idx_x[...,i])].astype(int))

    idx_y = insert_observation_fn(jnp.asarray(sensors_empty),jnp.asarray(idx_y))[0,...]
    sensors_loc_y = []
    for i in range(idx_y.shape[-1]):
        sensors_loc_y.append(idx_y[...,i][~np.isnan(idx_y[...,i])].astype(int))
    
    return [sensors_loc_x, sensors_loc_y]