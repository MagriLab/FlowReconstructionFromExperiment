import jax
import jax.numpy as jnp
import numpy as np
import chex
from typing import Union, Optional, Callable
import logging
logger = logging.getLogger(f'fr.{__name__}')

from .data import DataMetadata

Array = Union[np.ndarray, jax.Array]
Scalar = Union[int,float]




@jax.tree_util.Partial(jax.jit,static_argnames=('ax'))
def derivative2(f:Array, h:Scalar, ax:int=0) -> Array:
    '''Second derivatives with second order central difference for interior points and second order forward/backward difference for boundaries.\n
    
    Arguments:\n
        f: array of values to differentiate.\n
        h: step size, only constant step is supported.\n
        ax: int. Which axis to take derivative.\n
    
    Returns:
        d2fdx2: array of second derivatives with the same shape as f.
    '''

    f = jnp.asarray(f)

    try:
        chex.assert_axis_dimension_gteq(f,ax,4) # 
    except AssertionError as err:
        logger.error('Not enough nodes in the selected axis for the numerical scheme used.')
        raise err

    
    # initialise empty array for output
    d2fdx2 = jnp.empty_like(f)
    
    # for slicing the input later
    slice1 = [slice(None)]*f.ndim
    slice2 = [slice(None)]*f.ndim
    slice3 = [slice(None)]*f.ndim
    slice4 = [slice(None)]*f.ndim

    # interior points
    # second order scheme used f(x+h), f(x) and f(x-h)
    # = (f(x+h) - 2f(x) + f(x-h)) / (h**2)
    slice1[ax] = slice(2,None)
    slice2[ax] = slice(1,-1)
    slice3[ax] = slice(None,-2)

    d2fdx2 = d2fdx2.at[tuple(slice2)].set((f[tuple(slice1)] - 2*f[tuple(slice2)] + f[tuple(slice3)]) / (h**2))

    # left boundary (x=0)
    # second order forward difference
    # = (2f(x) - 5f(x+h) + 4f(x+2h) - f(x+3h)) / (h**2)
    slice1[ax] = slice(0,1)
    slice2[ax] = slice(1,2)
    slice3[ax] = slice(2,3)
    slice4[ax] = slice(3,4)

    d2fdx2 = d2fdx2.at[tuple(slice1)].set((2*f[tuple(slice1)] - 5*f[tuple(slice2)] + 4*f[tuple(slice3)] - f[tuple(slice4)]) / (h**2)) 
    
    # right boundary (x=L)
    # second order backward difference
    # = (2f(x) - 5f(x-h) + 4f(x-2h) - f(x-3h)) / (h**2)
    slice1[ax] = slice(-1,None)
    slice2[ax] = slice(-2,-1)
    slice3[ax] = slice(-3,-2)
    slice4[ax] = slice(-4,-3)

    d2fdx2 = d2fdx2.at[tuple(slice1)].set((2*f[tuple(slice1)] - 5*f[tuple(slice2)] + 4*f[tuple(slice3)] - f[tuple(slice4)]) / (h**2)) 

    return d2fdx2



@jax.tree_util.Partial(jax.jit,static_argnames=('ax'))
def derivative1(f:Array, h:Scalar, ax:int=0) -> Array:
    '''Second derivatives with second order central difference for interior points and second order forward/backward difference for boundaries.\n
    
    Arguments:\n
        f: array of values to differentiate.\n
        h: step size, only constant step is supported.\n
        ax: int. Which axis to take derivative.\n
    
    Returns:
        d2fdx2: array of second derivatives with the same shape as f.
    '''

    f = jnp.asarray(f)

    try:
        chex.assert_axis_dimension_gteq(f,ax,3) # 
    except AssertionError as err:
        logger.error('Not enough nodes in the selected axis for the numerical scheme used.')
        raise err

    # initialise empty array for output
    dfdx = jnp.empty_like(f)

    # for slicing the input later
    slice1 = [slice(None)]*f.ndim
    slice2 = [slice(None)]*f.ndim
    slice3 = [slice(None)]*f.ndim

    # interior points, central difference second order
    # = (f(x+h) - f(x-h)) / 2h
    slice1[ax] = slice(2,None)
    slice2[ax] = slice(1,-1)
    slice3[ax] = slice(None,-2)

    dfdx = dfdx.at[tuple(slice2)].set(
        (f[tuple(slice1)] - f[tuple(slice3)]) / (2*h)
    )

    # left boundary, second order forward difference
    # = (-3f(x) + 4f(x+h) - f(x+2h)) / 2h
    slice1[ax] = slice(0,1)
    slice2[ax] = slice(1,2)
    slice3[ax] = slice(2,3)

    dfdx = dfdx.at[tuple(slice1)].set(
        (-3*f[tuple(slice1)] + 4*f[tuple(slice2)] - f[tuple(slice3)]) / (2*h)
    )

    # right boundary, second order backward difference
    # = ( 3f(x) - 4f(x+h) + f(x+2h)) / 2h
    slice1[ax] = slice(-1,None)
    slice2[ax] = slice(-2,-1)
    slice3[ax] = slice(-3,-2)

    dfdx = dfdx.at[tuple(slice1)].set(
        (3*f[tuple(slice1)] - 4*f[tuple(slice2)] + f[tuple(slice3)]) / (2*h)
    )

    return dfdx



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

    if (not datainfo.problem_2d) and (uz is None):
        raise ValueError('Missing uz for 3D problem.')
    
    # set up which velocity 


    ## jax.lax.switch replace if



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
