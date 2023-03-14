import jax
import jax.numpy as jnp
import numpy as np
import chex
from typing import Union, Optional, Callable
import logging
logger = logging.getLogger(f'fr.{__name__}')

from .data import DataMetadata
from ._typing import Array, Scalar, ClassDataMetadata



@jax.tree_util.Partial(jax.jit,static_argnames=('axis'))
def derivative2(f:Array, h:Scalar, axis:int=0) -> Array:
    '''Second derivatives with second order central difference for interior points and second order forward/backward difference for boundaries.\n
    
    Arguments:\n
        f: array of values to differentiate.\n
        h: step size, only constant step is supported.\n
        axis: int. Which axis to take derivative.\n
    
    Returns:
        d2fdx2: array of second derivatives with the same shape as f.
    '''

    f = jnp.asarray(f)

    try:
        chex.assert_axis_dimension_gteq(f,axis,4) # 
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
    slice1[axis] = slice(2,None)
    slice2[axis] = slice(1,-1)
    slice3[axis] = slice(None,-2)

    d2fdx2 = d2fdx2.at[tuple(slice2)].set((f[tuple(slice1)] - 2*f[tuple(slice2)] + f[tuple(slice3)]) / (h**2))

    # left boundary (x=0)
    # second order forward difference
    # = (2f(x) - 5f(x+h) + 4f(x+2h) - f(x+3h)) / (h**2)
    slice1[axis] = slice(0,1)
    slice2[axis] = slice(1,2)
    slice3[axis] = slice(2,3)
    slice4[axis] = slice(3,4)

    d2fdx2 = d2fdx2.at[tuple(slice1)].set((2*f[tuple(slice1)] - 5*f[tuple(slice2)] + 4*f[tuple(slice3)] - f[tuple(slice4)]) / (h**2)) 
    
    # right boundary (x=L)
    # second order backward difference
    # = (2f(x) - 5f(x-h) + 4f(x-2h) - f(x-3h)) / (h**2)
    slice1[axis] = slice(-1,None)
    slice2[axis] = slice(-2,-1)
    slice3[axis] = slice(-3,-2)
    slice4[axis] = slice(-4,-3)

    d2fdx2 = d2fdx2.at[tuple(slice1)].set((2*f[tuple(slice1)] - 5*f[tuple(slice2)] + 4*f[tuple(slice3)] - f[tuple(slice4)]) / (h**2)) 

    return d2fdx2



@jax.tree_util.Partial(jax.jit,static_argnames=('axis'))
def derivative1(f:Array, h:Scalar, axis:int=0) -> Array:
    '''First derivatives with second order central difference for interior points and second order forward/backward difference for boundaries.\n
    
    Arguments:\n
        f: array of values to differentiate.\n
        h: step size, only constant step is supported.\n
        axis: int. Which axis to take derivative.\n
    
    Returns:
        dfdx: array of 1st derivatives with the same shape as f.
    '''

    f = jnp.asarray(f)

    try:
        chex.assert_axis_dimension_gteq(f,axis,3) # 
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
    slice1[axis] = slice(2,None)
    slice2[axis] = slice(1,-1)
    slice3[axis] = slice(None,-2)

    dfdx = dfdx.at[tuple(slice2)].set(
        (f[tuple(slice1)] - f[tuple(slice3)]) / (2*h)
    )

    # left boundary, second order forward difference
    # = (-3f(x) + 4f(x+h) - f(x+2h)) / 2h
    slice1[axis] = slice(0,1)
    slice2[axis] = slice(1,2)
    slice3[axis] = slice(2,3)

    dfdx = dfdx.at[tuple(slice1)].set(
        (-3*f[tuple(slice1)] + 4*f[tuple(slice2)] - f[tuple(slice3)]) / (2*h)
    )

    # right boundary, second order backward difference
    # = ( 3f(x) - 4f(x+h) + f(x+2h)) / 2h
    slice1[axis] = slice(-1,None)
    slice2[axis] = slice(-2,-1)
    slice3[axis] = slice(-3,-2)

    dfdx = dfdx.at[tuple(slice1)].set(
        (3*f[tuple(slice1)] - 4*f[tuple(slice2)] + f[tuple(slice3)]) / (2*h)
    )

    return dfdx



def div_field(
    ux:Array,
    uy:Array,
    datainfo:ClassDataMetadata,
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
    
    dudx = derivative1(ux,datainfo.dx,axis=datainfo.axx)
    dvdy = derivative1(uy,datainfo.dy,axis=datainfo.axy)
    div = dudx + dvdy
    if uz is not None:
        dwdz = derivative1(uz,datainfo.dz,axis=datainfo.axz)
        div = div + dwdz
        logger.info('Divergence calculated for 3D flow.')
    return div


def momentum_residue_field(
    ux:Array,
    uy:Array,
    p:Array,
    datainfo:ClassDataMetadata,
    uz:Optional[Array] = None,
    **kwargs) -> Array:
    
    '''Calculate the momentum residue of the nondimensional Navier Stokes equation for selected velocity.
    Either 2D or 3D.\n
    
    Arguments:\n
        ux: array of velocity in x direction.\n
        uy: array of velocity in y direction.\n
        p: array of pressure.\n
        datainfo: an instance of DataMetadata.\n
        uz: array of velocity in z direction.\n

    return:\n
        Momentum residue field, has shape [i,...], where i is the number of velocity and ... is the shape of the input velocity field (e.g. ux.shape).
    '''

    try:
        chex.assert_equal_shape([ux,uy,p])
        u = jnp.stack((ux,uy),axis=0)
        if uz is not None:
            chex.assert_equal_shape([ux,uz])
            u = jnp.concatenate((u,uz[jnp.newaxis,...]),axis=0)
    except AssertionError as err:
        logger.error('Cannot calculate momentum residue, input shape mismatch.')
        raise err
    try:
        chex.assert_rank(u,[2+u.shape[0]])
    except AssertionError as err:
        logger.error(f'Cannot calculate momentum, number of velocities does not match the number of dimensions.')
        raise err

    
    step_space = datainfo.discretisation[1:]
    axis_space = datainfo.axis_index[1:]
    

    ## Define internal functions
    v_derivative1 = jax.vmap(derivative1,(0,None,None),0)
    v_derivative2 = jax.vmap(derivative2,(0,None,None),0)
    
    # function that applies a function to inn, and x,y,z in order
    def _didj(de_fun,inn):
        didj_T = de_fun(inn,datainfo.dx,datainfo.axx).reshape((-1,)+inn.shape)
        for i in range (1,u.shape[0]):
            didj_T = jnp.concatenate(
                (
                didj_T,
                de_fun(inn,step_space[i],axis_space[i]).reshape((-1,)+inn.shape)
                ),
                axis=0
            )
        return didj_T # for de_fun = v_derivative1 and inn=u -> [j,i,t,x,y,z]
    

    ## calculate derivatives
    dui_dt = v_derivative1(u,datainfo.dt,datainfo.axt) # [i,t,x,y,z]

    # # output convection terms
    # (u*du/dx + v*du/dy,
    #  u*dv/dx + v*dv/dy)
    dui_dxj_T = _didj(v_derivative1,u)
    ududx_i = jnp.einsum('j..., ji... -> i...', u, dui_dxj_T) # [i,t,x,y,z]

    dpdx_i = _didj(derivative1,p) #[i,t,x,y,z]

    # # output second derivatives
    # ((d2u/dx2, d2u/dy2),
    #  (d2v/dx2, d2v/dy2))
    d2ui_dxj2 = _didj(v_derivative2,u)
    d2udx2_i = jnp.einsum('ji... - > i...', d2ui_dxj2) # [i,t,x,y,z]

    residue = dui_dt + ududx_i + dpdx_i - (d2udx2_i/datainfo.re)
    
    return residue
