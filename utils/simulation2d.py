import numpy as np
import h5py
from pathlib import Path
import warnings
from typing import Union, Optional

import logging

path = Union[str,Path]
number = Union[float, int, np.number]

logger = logging.getLogger(f'fr.{__name__}')
logger.warning('warning')
logger.info('info')

def read_data(dir:path, idx_body:int):
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