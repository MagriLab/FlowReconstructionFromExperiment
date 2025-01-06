import numpy as np
import h5py
import warnings
import logging
from pathlib import Path
from typing import Union, Optional


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
    shape = [1]*dim
    shape[1] = -1
    f = np.broadcast_to(f_single_line.reshape(tuple(shape)), [ngrid]*dim) # expand in all dimension but y. 
    f = f.reshape((1,)+f.shape)
    zeros = np.zeros((dim-1,)+f.shape)
    f = f.reshape((1,)+ f.shape)
    f = np.concatenate([f,zeros],axis=0)
    return -f



# ======================= Volvo =================================

def read_data_volvo(data_path: path, nondimensional = True):
    """Read the volvorig data."""

    data_path = Path(data_path)
    if not data_path.exists():
        raise ValueError(f"Data path '{data_path.absolute()}' does not exist.")

    data_dict = np.load(Path(data_path, 'data.npz'))
    u_p = data_dict['data']
    density = data_dict['density']
    x = data_dict['x']
    dx = x[1]-x[0]
    y = data_dict['y']
    dy = y[1]-y[0]
    z = data_dict['z']
    dz = z[1]-z[0]
    t = data_dict['t']
    dt = t[1]-t[0]

    # for non dimensionalisation
    l0 = 0.04 #m
    p0 = 101325 #air
    density = density.mean()
    mu = 17.88e-6
    viscosity = mu/density

    if data_path.name == 'u166':
        u0 = 16.6 #m/s
    
    re = u0*l0/viscosity
    t0 = l0/u0

    if nondimensional:
        u_p[...,:-1] = u_p[...,:-1]/u0
        u_p[...,-1] = u_p[...,-1]/p0

        return u_p, [dt/t0, dx/l0, dy/l0, dz/l0], re, density
    else:
        return u_p, [dt, dx, dy, dz], re, density
