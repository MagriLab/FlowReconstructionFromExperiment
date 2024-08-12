import numpy as np
import h5py
import warnings
import logging
import jax.numpy as jnp
from pathlib import Path
from typing import Union, Optional
from scipy.interpolate import RBFInterpolator


path = Union[str,Path]
number = Union[float, int, np.number]

logger = logging.getLogger(f'fr.{__name__}')


# ================== 2D triangle ==============================

def read_data_2dtriangle(dir:path, idx_body:int):
    '''Read the data for 2d triangle bluff body simulation.
    
    Arguments:\n
        dir: path to the data files. \n
        idx_body: the x index of the base of the body, data before this index is discarded.\n
    Return:\n
        (ux,uy,pp): data
    '''
    with h5py.File(Path(dir,"ux.h5"),'r') as hf:
        ux = np.array(hf.get("ux"))
    with h5py.File(Path(dir,"uy.h5"),'r') as hf:
        uy = np.array(hf.get("uy"))
    with h5py.File(Path(dir,"pp.h5"),'r') as hf:
        pp = np.array(hf.get("pp"))
        
    ux = np.delete(ux,np.s_[:idx_body],1)
    uy = np.delete(uy,np.s_[:idx_body],1)
    pp = np.delete(pp,np.s_[:idx_body],1)
    
    return (ux,uy,pp)


def take_measurement_base(data:np.ndarray, 
                            ly:number, 
                            centrex:number, 
                            domain_y:Optional[number] = None,
                            domain_x:Optional[number] = None):
    '''Take a line measurement at the base of the body.
    
    Arguments:\n
        data: the measurements of the entire domain. Last two dimensions must be x, y.\n
        ly: length of the base (across y direction). Given as 2 indices [start, end] or two positive numbers with units [10,20] in meters.\n
        centrex: x coord of the centre of the base. Given as index or with positive number.
        For example, if the base is located 2m from inlet, centrex=2..\n
        domain_y: total length of domain in the same unit as ly, only needed if ly has units.\n
        domain_x: total length of domain in the same unit as centrex, only needed if centrex has units.\n
    '''

    if data.ndim < 2:
        raise ValueError("Data is not a 2D flow field.")

    nx = data.shape[-2]
    ny = data.shape[-1]

    if domain_y is not None:
        if (ly[0] < 0) or (ly[1] > domain_y):
            warnings.warn("Length of the body is out of the given domain width, ignoring argument 'domain_y'")
            domain_y = None

    if domain_x is not None:
        if (centrex>domain_x) or (centrex<0):
            warnings.warn("Body is located outside the given domain length, ignoring argument 'domain_x'")
            domain_x = None


    if domain_x is None:
        idx_x = int(centrex)
    else:
        idx_x = int(np.ceil(nx*centrex/domain_x))
        logger.info(f'Calculating index. Taking the first available measurement behind the body. Index {idx_x}.')
    
    idx_y = [0,0] 
    if domain_y is None:
        idx_y = ly
    else:
        idx_y[0] = int(np.ceil(ny*ly[0]/domain_y))
        idx_y[1] = int(np.floor(ny*ly[1]/domain_y))
        logger.info(f'Calculating index. Measurements start at the closest available points that are inside the width of the body. Index {idx_y}.')

    return data[...,idx_x,np.s_[idx_y[0]:idx_y[1]]]



def interpolate_2dtriangle(u, pb, case_observe, datacfg):
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
            rbf = RBFInterpolator(sensors_loc,temp_measurement.flatten(),kernel='thin_plate_spline')
            _interp = rbf(_locs).reshape(grid_x.shape)
            compare_interp.append(_interp)
    compare_interp = np.array(compare_interp)
    compare_interp = np.stack((compare_interp[:nt,...],compare_interp[nt:2*nt,...],compare_interp[2*nt:3*nt,...]),axis=-1)

    return compare_interp, temp_observed



# ===================== Kolmogorov flow =========================

def read_data_kolsol(data_path: path):
    ''' Read Kolmogorov flow data generated using KolSol.\n
    Returns data with shape [t, x, y, ..., dim+1], the last dimension contains u1, u2, ..., p.
    '''

    data_path = Path(data_path)
    if not data_path.exists():
        raise ValueError(f"Data path '{data_path.absolute()}' does not exist.")

    with h5py.File(data_path) as hf:
        u_p = np.array(hf.get('state'))
        dt = float(hf.get('dt')[()])
        re = float(hf.get('re')[()])
    
    return u_p, re, dt

def kolsol_forcing_term(k:float, ngrid: int, dim:int) -> np.array:
    x = np.linspace(0, 2*np.pi, ngrid+1)[:-1]
    f_single_line = np.sin(k*x)
    f = np.tile(f_single_line.reshape((1,-1)),[ngrid,1])
    f = np.stack([f,np.zeros_like(f)],axis=0).reshape((dim,1,ngrid,ngrid))
    return -f




class Interpolator():
    def __init__():
        pass