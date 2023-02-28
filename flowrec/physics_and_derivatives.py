import jax.numpy as jnp
import numpy as np
import chex
from typing import Union, Optional, Callable
import logging
logger = logging.getLogger(f'fr.{__name__}')

Array = Union[np.ndarray, jnp.ndarray]
Scalar = Union[int,float]

def div_field(
    ux:Array,
    uy:Array,
    dx:Scalar,
    dy:Scalar,
    x_axis:int,
    y_axis:int,
    uz:Optional[Array]=None,
    dz:Optional[Scalar]=None,
    z_axis:Optional[int]=None) -> Array:
    '''Calculate the diveregence of the flow given some snapshots.\n
    
    Arguments:\n
        ux: an array of velocity in x direction.\n
        uy: an array of velocity in x direction.\n
        dx: space in the x direction between each value in ux.\n
        dy: space in the y direction between each value in uy.\n
        x_axis: which axis is x axis.\n
        y_axis: which axis is y axis.\n
        uz: an array of velocity in z direction, None if the 2D flow.\n
        dz: space in the z direction between each value in uz, None if the 2D flow.\n
        z_axis: which axis is z axis, None if the 2D flow.\n

    Return:\n
        An array of divergence of the flow with the same as ux, uy and uz.
    '''
    if uz:
        chex.assert_equal_shape((ux,uy,uz))
    else:
        chex.assert_equal_shape((ux,uy))
    
    dudx = jnp.gradient(ux,dx,axis=x_axis)
    dvdy = jnp.gradient(uy,dy,axis=y_axis)
    div = dudx + dvdy
    if uz:
        dwdz = jnp.gradient(uz,dz,axis=z_axis)
        div = div + dwdz
        logger.info('Divergence calculated for 3D flow.')
    return div


def momentum_residue_field(
    which_velocity:int,
    ux:Array,
    uy:Array,
    p:Array,
    re:Scalar,
    dt:Scalar,
    dx:Scalar,
    dy:Scalar,
    axis_idx:list, # [t,x,y,z]
    **kwargs) -> Array:
    
    '''Calculate the momentum residue of the nondimensional Navier Stokes equation for selected velocity.
    Either 2D or 3D.\n
    
    Arguments:\n
        which_velocity: either 1, 2 or 3 to calculate momentum about ux, uy or uz.\n
        ux: array of velocity in x direction.\n
        uy: array of velocity in y direction.\n
        p: array of pressure.\n
        re: Reynolds number.\n
        dt: time between snapshots.\n
        dx: distance between each measured value in x direction.\n
        dy: distance between each measured value in x direction.\n
        axis_idx: list of index indicating the axis for [t,x,y] or [t,x,y,z].\n
        **kwargs: if 3D, provide 'uz' and 'dz'\n
    
    return:\n
        Momentum residue with the same shape as input flow field.
    '''

    chex.assert_equal_shape((ux,uy,p))
    
    if not (jnp.isscalar(re) and jnp.isscalar(dt) and jnp.isscalar(dx) and jnp.isscalar(dy)):
        raise ValueError('Reynolds number, dt, dx and dy must be scalars.')

    if len(axis_idx) == 3:
        axt,axx,axy = axis_idx
    elif len(axis_idx) == 4:
        axt,axx,axy,axz = axis_idx
    else:
        raise ValueError('Momentum residue can only calculate 2D or 3D mometum.')
    
    # set up which velocity 
    if which_velocity == 1:
        u = ux
        axp = axx
        p_dx = dx
        logger.info('Momentum calculated for ux.')
    elif which_velocity == 2:
        u = uy
        axp = axy
        p_dx = dy
        logger.info('Momentum calculated for uy.')
    elif which_velocity == 3:
        if ('uz' in kwargs) and ('dz' in kwargs) and ('axz' in locals()):
            uz = kwargs['uz']
            dz = kwargs['dz']
        else:
            raise ValueError('Missing uz, dz, or index for z axis.')
        u = uz
        axp = axz
        p_dx = dz
        logger.info('Momentum calculated for uz.')
    else:
        raise ValueError('Please set "which_velocity" to 1/2/3 for ux/uy.uz')

    
    dudt = jnp.gradient(u,dt,axis=axt)

    dudx = jnp.gradient(u,dx,axis=axx)
    d2udx2 = jnp.gradient(dudx,dx,axis=axx)
    ux_dudx = ux*dudx

    dudy = jnp.gradient(u,dy,axis=axy)
    d2udy2 = jnp.gradient(dudy,dy,axis=axy)
    uy_dudy = uy*dudy

    dp = jnp.gradient(p,p_dx,axis=axp)

    residue = dudt + ux_dudx + uy_dudy + dp - (d2udx2 + d2udy2)/re

    if which_velocity == 3:
        dudz = jnp.gradient(u,dz,axis=axz)
        d2udz2 = jnp.gradient(dudz,dz,axis=axz)
        uz_dudz = uz*dudz
        residue = residue + uz_dudz - d2udz2/re
    
    return residue
