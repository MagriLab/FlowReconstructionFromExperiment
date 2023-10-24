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

    interior = (f[tuple(slice1)] - 2*f[tuple(slice2)] + f[tuple(slice3)]) / (h**2)

    # left boundary (x=0)
    # second order forward difference
    # = (2f(x) - 5f(x+h) + 4f(x+2h) - f(x+3h)) / (h**2)
    slice1[axis] = slice(0,1)
    slice2[axis] = slice(1,2)
    slice3[axis] = slice(2,3)
    slice4[axis] = slice(3,4)

    left = (2*f[tuple(slice1)] - 5*f[tuple(slice2)] + 4*f[tuple(slice3)] - f[tuple(slice4)]) / (h**2)
    
    # right boundary (x=L)
    # second order backward difference
    # = (2f(x) - 5f(x-h) + 4f(x-2h) - f(x-3h)) / (h**2)
    slice1[axis] = slice(-1,None)
    slice2[axis] = slice(-2,-1)
    slice3[axis] = slice(-3,-2)
    slice4[axis] = slice(-4,-3)

    right = (2*f[tuple(slice1)] - 5*f[tuple(slice2)] + 4*f[tuple(slice3)] - f[tuple(slice4)]) / (h**2)

    return jnp.concatenate((left,interior,right),axis=axis)



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

    # for slicing the input later
    slice1 = [slice(None)]*f.ndim
    slice2 = [slice(None)]*f.ndim
    slice3 = [slice(None)]*f.ndim

    # interior points, central difference second order
    # = (f(x+h) - f(x-h)) / 2h
    slice1[axis] = slice(2,None)
    slice2[axis] = slice(1,-1)
    slice3[axis] = slice(None,-2)

    interior = (f[tuple(slice1)] - f[tuple(slice3)]) / (2*h)

    # left boundary, second order forward difference
    # = (-3f(x) + 4f(x+h) - f(x+2h)) / 2h
    slice1[axis] = slice(0,1)
    slice2[axis] = slice(1,2)
    slice3[axis] = slice(2,3)

    left = (-3*f[tuple(slice1)] + 4*f[tuple(slice2)] - f[tuple(slice3)]) / (2*h)

    # right boundary, second order backward difference
    # = ( 3f(x) - 4f(x+h) + f(x+2h)) / 2h
    slice1[axis] = slice(-1,None)
    slice2[axis] = slice(-2,-1)
    slice3[axis] = slice(-3,-2)

    right = (3*f[tuple(slice1)] - 4*f[tuple(slice2)] + f[tuple(slice3)]) / (2*h)

    return jnp.concatenate((left,interior,right),axis=axis)



def div_field(
    u:Array,
    datainfo:ClassDataMetadata) -> Array:
    '''Calculate the diveregence of the flow given some snapshots.\n
    
    Arguments:\n
        u: an array of velocities with shape [t,x,y,...,i], i=2 if 2D flow, 3 if 3D flow. \n
        datainfo: an instance of DataMetadata.\n

    Return:\n
        An array of divergence of the flow with shape [t,x,y].
    '''
    step_space = datainfo.discretisation[1:]
    axis_space = datainfo.axis_index[1:]
    
    chex.assert_axis_dimension(u,-1,len(step_space))

    div = derivative1(u[...,0],step_space[0],axis_space[0])
    for i in range(1,u.shape[-1]):
        div = div + derivative1(u[...,i],step_space[i],axis_space[i])

    return div


def momentum_residual_field(
    u_p:Array,
    datainfo:ClassDataMetadata,
    **kwargs) -> Array:
    
    '''Calculate the momentum residue of the nondimensional Navier Stokes equation for selected velocity.
    Either 2D or 3D, no forcing term or gravity\n
    
    Arguments:\n
        u_p: array of velocitie and pressure with shape [t,x,y,...,i], u_p[...,-1] is the pressure field. \n
        datainfo: an instance of DataMetadata.\n

    return:\n
        Momentum residue field, has shape [i,...], where i is the number of velocity and ... is the shape of the input velocity field.
    '''

    
    step_space = datainfo.discretisation[1:]
    axis_space = datainfo.axis_index[1:]

    chex.assert_axis_dimension(u_p,-1,len(step_space)+1)

    u = jnp.moveaxis(u_p[...,:-1],-1,0)
    p = u_p[...,-1]
    

    ## Define internal functions
    v_derivative1 = jax.vmap(derivative1,(0,None,None),0)
    v_derivative2 = jax.vmap(derivative2,(0,None,None),0)
    
    # function that applies a function to inn, and x,y,z in order
    def _didj(de_fun,inn):
        didj_T = de_fun(inn,datainfo.dx,datainfo.axx).reshape((-1,)+inn.shape)
        for j in range (1,u.shape[0]):
            didj_T = jnp.concatenate(
                (
                didj_T,
                de_fun(inn,step_space[j],axis_space[j]).reshape((-1,)+inn.shape)
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

    residual = dui_dt + ududx_i + dpdx_i - (d2udx2_i/datainfo.re)
    
    return residual
