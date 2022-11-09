import numpy as np
from typing import Optional, Union
import sys

dtype = Union[str,np.dtype]

def data_partition(data:np.ndarray,
                    axis:int,
                    partition:list,
                    SHUFFLE:bool=True,
                    REMOVE_MEAN:bool=False,
                    data_type:dtype=np.float32,
                    rng:Optional[np.random.Generator]=None) -> tuple[np.ndarray]: 
    '''Split the data into sets.
    
    Arguments:\n
        data: a numpy array of data\n
        axis: which axis to split\n
        partition: a list giving the number of snapshots for each set\n
        SHUFFLE: if true the data is shuffled. Default true.\n
        REMOVE_MEAN: if true the data is centered around 0. Default false.\n
        data_type: default numpy.float32.\n
        rng: a numpy random number generator\n
    Return:\n
        A tuple of numpy arrays
    '''
    if np.sum(partition) > data.shape[axis]:
        raise ValueError("Not enough snapshots in the dataset.")
    d = np.copy(data) # shuffle is an in-place operation

    d = np.moveaxis(d,axis,0)

    if SHUFFLE:
        if rng is not None:
            rng.shuffle(d)
        else:
            np.random.shuffle(d)

    # split into sets
    datasets = [np.moveaxis(d[:partition[0],...],0,axis).astype(data_type)]
    for i in range(1,len(partition)):
        if REMOVE_MEAN:
            a = d[partition[i-1]:(partition[i]+partition[i-1]),...] - np.mean(d[partition[i-1]:partition[i],...],axis=0)
            a = np.moveaxis(a,0,axis)
            datasets.append(np.moveaxis(a,0,axis).astype(data_type))
        else:
            a = d[partition[i-1]:(partition[i]+partition[i-1]),...]
            a = np.moveaxis(a,0,axis)
            datasets.append(a.astype(data_type))
    
    return datasets