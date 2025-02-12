import jax
import jax.numpy as jnp
import numpy as np
import chex
from typing import Union, Optional, Callable, Tuple
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
    slice5 = [slice(None)]*f.ndim

    # # interior points
    # # second order scheme used f(x+h), f(x) and f(x-h)
    # # = (f(x+h) - 2f(x) + f(x-h)) / (h**2)
    # slice1[axis] = slice(2,None)
    # slice2[axis] = slice(1,-1)
    # slice3[axis] = slice(None,-2)

    # interior = (f[tuple(slice1)] - 2*f[tuple(slice2)] + f[tuple(slice3)]) / (h**2)

    # left boundary (x=0)
    # second order forward difference
    # = (2f(x) - 5f(x+h) + 4f(x+2h) - f(x+3h)) / (h**2)
    slice1[axis] = slice(0,2)
    slice2[axis] = slice(1,3)
    slice3[axis] = slice(2,4)
    slice4[axis] = slice(3,5)

    left = (2*f[tuple(slice1)] - 5*f[tuple(slice2)] + 4*f[tuple(slice3)] - f[tuple(slice4)]) / (h**2)
    
    # right boundary (x=L)
    # second order backward difference
    # = (2f(x) - 5f(x-h) + 4f(x-2h) - f(x-3h)) / (h**2)
    slice1[axis] = slice(-2,None)
    slice2[axis] = slice(-3,-1)
    slice3[axis] = slice(-4,-2)
    slice4[axis] = slice(-5,-3)

    right = (2*f[tuple(slice1)] - 5*f[tuple(slice2)] + 4*f[tuple(slice3)] - f[tuple(slice4)]) / (h**2)

    # 4th order interior points
    # = (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) -f(x-2h)) / (12 * h**2)
    slice1[axis] = slice(4,None)
    slice2[axis] = slice(3,-1)
    slice3[axis] = slice(2,-2)
    slice4[axis] = slice(1,-3)
    slice5[axis] = slice(None,-4)

    interior = (-f[tuple(slice1)] + 16*f[tuple(slice2)] - 30*f[tuple(slice3)] + 16*f[tuple(slice4)] - f[tuple(slice5)]) / (12* (h**2))

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
    slice4 = [slice(None)]*f.ndim

    # # interior points, central difference second order
    # # = (f(x+h) - f(x-h)) / 2h
    # slice1[axis] = slice(2,None)
    # slice2[axis] = slice(1,-1)
    # slice3[axis] = slice(None,-2)

    # interior = (f[tuple(slice1)] - f[tuple(slice3)]) / (2*h)

    # left boundary, second order forward difference
    # = (-3f(x) + 4f(x+h) - f(x+2h)) / 2h
    slice1[axis] = slice(0,2)
    slice2[axis] = slice(1,3)
    slice3[axis] = slice(2,4)

    left = (-3*f[tuple(slice1)] + 4*f[tuple(slice2)] - f[tuple(slice3)]) / (2*h)

    # right boundary, second order backward difference
    # = ( 3f(x) - 4f(x+h) + f(x+2h)) / 2h
    slice1[axis] = slice(-2,None)
    slice2[axis] = slice(-3,-1)
    slice3[axis] = slice(-4,-2)

    right = (3*f[tuple(slice1)] - 4*f[tuple(slice2)] + f[tuple(slice3)]) / (2*h)

    # 4th order central difference interior points
    # = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h
    slice1[axis] = slice(4,None)
    slice2[axis] = slice(3,-1)
    slice3[axis] = slice(1,-3)
    slice4[axis] = slice(None,-4)

    interior = ( -f[tuple(slice1)] + 8*f[tuple(slice2)] - 8*f[tuple(slice3)] + f[tuple(slice4)]) / (12*h)

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
    Either 2D or 3D.\n
    
    Arguments:\n
        u_p: array of velocitie and pressure with shape [t,x,y,...,i], u_p[...,-1] is the pressure field. \n
        datainfo: an instance of DataMetadata.\n

    Available kwargs:\n
        forcing: array of forcing term. The residual will return as dudt + u*dudx + dpdx - (re^-1) * d2udx2 + forcing 

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

    if 'forcing' in kwargs and (kwargs['forcing'] is not None):
        logger.debug('Forced flow')
        return residual + kwargs['forcing']
    
    return residual



def dissipation(
    u: Array, 
    datainfo: ClassDataMetadata
    ) -> Array:
    """Calculate dissipation from velocity for non-dimensional flows viscosity = 1/Re.
    
    Arguments:\n
        u: [...,u], last dimension must be velocity.\n
        datainfo: an instance of DataMetadata.\n
    
    Return:\n
        dissipation: of shape [t,x,y,z]
    """
    logger.warning('Dissipation not normalised.')
    # the calculated dissipation matches Elise's results in shape, but not magnitude. This dissipation is not normalised.

    step_space = datainfo.discretisation[1:]
    axis_space = datainfo.axis_index[1:]
    re = datainfo.re
    
    # check dimension and prepare matrix
    u = jnp.moveaxis(u[...],-1,0) # move velocity axis to 0
    chex.assert_axis_dimension(u,0,len(step_space))

    v_derivative1 = jax.vmap(derivative1,(0,None,None),0)
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
    
    dui_dxj_T = _didj(v_derivative1, u) # [j,i,t,x,y,z]
    dijsumdji = jnp.einsum('ji... -> ij...', dui_dxj_T) + dui_dxj_T # [i,j,t,x,y,z]
    d = jnp.einsum('ijt... -> t...', dijsumdji**2) / re # [t,x,y,z]

    return d
 


@jax.tree_util.Partial(jax.jit,static_argnames=('datainfo'))
def vorticity(u:Array, datainfo:ClassDataMetadata) -> Array:
    """Calculate vorticity field.\n
    
    Arguments:\n
        u: [...,u], last dimension must be velocity.\n
        datainfo: an instance of DataMetadata.\n
    
    Return:\n
        vorticity: of shape [1,x,y,z]
    """

    step_space = datainfo.discretisation[1:]
    axis_space = datainfo.axis_index[1:]
    u = jnp.moveaxis(u[...],-1,0) # move velocity axis to 0
    v_derivative1 = jax.vmap(derivative1,(0,None,None),0)
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
    
    dui_dxj = jnp.einsum('jit... -> ijt...', _didj(v_derivative1, u)) # [i,j,t,x,y,z]

    vort = dui_dxj[1,0,...] - dui_dxj[0,1,...]
    if len(axis_space) == 2:
        return vort
    elif len(axis_space) == 3:
        vort2 = vort
        vort0 = dui_dxj[2,1,...] - dui_dxj[1,2,...]
        vort1 = dui_dxj[0,2,...] - dui_dxj[2,0,...]
        return jnp.stack([vort0,vort1,vort2], )
    else:
        raise ValueError



def extreme_dissipation_threshold(global_dissipation: Array):
    """
    Calculate the threshold of extreme events.
    -----------------------------------------
    *Farazmand, Mohammad, and Themistoklis P. Sapsis. ‘A Variational Approach to Probing Extreme Events in Turbulent Dynamical Systems’. Science Advances, vol. 3, no. 9, Sept. 2017, p. e1701533. DOI.org (Crossref), https://doi.org/10.1126/sciadv.1701533.*
    """
    return np.mean(global_dissipation) + 2*np.std(global_dissipation)


def count_extreme_events(global_dissipation: Array, threshold: Scalar) -> Tuple[list, int]:
    """Count the number of extreme events
    -------------------------- 
    Count the number time the global dissipation passes the threshold of extreme events
    ### Arguments:
        - Global dissipation: 1D array of volume averaged dissipation over time.
        - Threshold: the value that defines an extreme event (mean(global_di)+2*std).
    ### Outputs:
        - Array indices of the start of every extreme events.
        - Number of extreme events.
    """
    binary_array = global_dissipation >= threshold
    start = binary_array[0]
    change_idx = list(np.where(np.diff(binary_array))[0])
    if start:
        event_start_idx = [0]
        event_start_idx.extend(change_idx[1::2]) # change_idx[0] is the end of an extreme event
    else:
        event_start_idx = change_idx[::2]
        
    return event_start_idx, len(event_start_idx)
    


def get_tke(ufluc:Array, datainfo:ClassDataMetadata, **kwargs) -> tuple[Array, Array, Array]:
    """Calculate the turbulent kinetic energy
    ==================================

    - ufluc: velocities, must be shape [t,x,y,u] or [t,x,y,z,u].
    - datainfo: an instance of DataMetadata.
    - Available kwargs: kgrid_magnitude

    return
    - spectrum: the turbulent kinetic energy sorted by wavenumber.
    - kbins: a length nbins array of corresponding wavenumbers 

    """
    _shape = np.array(ufluc.shape)
    nx = _shape[1:-1]
    dx = datainfo.discretisation[1:]
    if 'kgrid_magnitude' in kwargs.keys():
        kgrid_magnitude_int = kwargs['kgrid_magnitude']
    else:
        fftfreq = []
        dk = 2*np.pi/np.array(dx)
        for i in range(len(nx)):
            _k = np.fft.fftfreq(nx[i])*dk[i]
            fftfreq.append(_k)
        
        kgrid = np.meshgrid(*fftfreq, indexing='ij')
        kgrid = np.array(kgrid)
        kgrid_magnitude = np.sqrt(np.einsum('n... -> ...', kgrid**2))
        kgrid_magnitude_int = kgrid_magnitude.astype('int')
    kmax = np.max(kgrid_magnitude_int)
    kbins = np.arange(kmax).astype('int')

    u_fft = np.fft.fftn(ufluc,axes=list(range(1,len(dx)+1)))
    ke_fft = np.sum(u_fft * np.conj(u_fft),axis=-1).real * 0.5
    ke_avg = np.mean(ke_fft,axis=0)
    spectrum = np.zeros_like(kbins).astype('float32')
    for i in kbins:
        spectrum[i] += 0.5*np.sum(ke_avg[kgrid_magnitude_int==i])
    
    return spectrum, kbins