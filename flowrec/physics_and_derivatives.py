import jax.numpy as jnp
import numpy as np
import chex
from typing import Union, Optional, Callable
import logging
logger = logging.getLogger(f'fr.{__name__}')

from .data import DataMetadata

Array = Union[np.ndarray, jnp.ndarray]
Scalar = Union[int,float]

def div_field(
    ux:Array,
    uy:Array,
    datainfo:DataMetadata,
    uz:Optional[Array]=None) -> Array:
    '''Calculate the diveregence of the flow given some snapshots.\n
    
    Arguments:\n
        ux: an array of velocity in x direction.\n
        uy: an array of velocity in x direction.\n
        datainfo: an instance of DataMetadata.\n
        uz: an array of velocity in z direction, None if the 2D flow.\n

    Return:\n
        An array of divergence of the flow with the same as ux, uy and uz.
    '''
    if uz is not None:
        chex.assert_equal_shape((ux,uy,uz))
    else:
        chex.assert_equal_shape((ux,uy))
    
    dudx = jnp.gradient(ux,datainfo.dx,axis=datainfo.axx)
    dvdy = jnp.gradient(uy,datainfo.dy,axis=datainfo.axy)
    div = dudx + dvdy
    if not datainfo.problem_2d:
        dwdz = jnp.gradient(uz,datainfo.dz,axis=datainfo.axz)
        div = div + dwdz
        logger.info('Divergence calculated for 3D flow.')
    return div


def momentum_residue_field(
    which_velocity:int,
    ux:Array,
    uy:Array,
    p:Array,
    datainfo:DataMetadata,
    uz:Optional[Array] = None,
    **kwargs) -> Array:
    
    '''Calculate the momentum residue of the nondimensional Navier Stokes equation for selected velocity.
    Either 2D or 3D.\n
    
    Arguments:\n
        which_velocity: either 1, 2 or 3 to calculate momentum about ux, uy or uz.\n
        ux: array of velocity in x direction.\n
        uy: array of velocity in y direction.\n
        p: array of pressure.\n
        datainfo: an instance of DataMetadata.\n
        uz: array of velocity in z direction.\n

    return:\n
        Momentum residue with the same shape as input flow field.
    '''

    chex.assert_equal_shape((ux,uy,p))

    if (not datainfo.problem_2d) and (not uz):
        raise ValueError('Missing uz for 3D problem.')
    
    # set up which velocity 
    if which_velocity == 1:
        u = ux
        axp = datainfo.axx
        p_dx = datainfo.dx
        logger.info('Momentum calculated for ux.')
    elif which_velocity == 2:
        u = uy
        axp = datainfo.axy
        p_dx = datainfo.dy
        logger.info('Momentum calculated for uy.')
    elif which_velocity == 3:
        u = uz
        axp = datainfo.axz
        p_dx = datainfo.dz
        logger.info('Momentum calculated for uz.')
    else:
        raise ValueError('Please set "which_velocity" to 1/2/3 for ux/uy/uz')

    
    dudt = jnp.gradient(u,datainfo.dt,axis=datainfo.axt)

    dudx = jnp.gradient(u,datainfo.dx,axis=datainfo.axx)
    d2udx2 = jnp.gradient(dudx,datainfo.dx,axis=datainfo.axx)
    ux_dudx = ux*dudx

    dudy = jnp.gradient(u,datainfo.dy,axis=datainfo.axy)
    d2udy2 = jnp.gradient(dudy,datainfo.dy,axis=datainfo.axy)
    uy_dudy = uy*dudy

    dp = jnp.gradient(p,p_dx,axis=axp)

    residue = dudt + ux_dudx + uy_dudy + dp - (d2udx2 + d2udy2)/datainfo.re

    if not datainfo.problem_2d:
        dudz = jnp.gradient(u,datainfo.dz,axis=datainfo.axz)
        d2udz2 = jnp.gradient(dudz,datainfo.dz,axis=datainfo.axz)
        uz_dudz = uz*dudz
        residue = residue + uz_dudz - d2udz2/datainfo.re
    
    return residue
