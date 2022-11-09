import numpy as np
import h5py
from pathlib import Path
from typing import Union

path = Union[str,Path]

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