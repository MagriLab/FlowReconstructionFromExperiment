import numpy as np
from typing import Optional, Union

dtype = Union[str,np.dtype]

def data_partition(data:np.ndarray,
                    axis:int,
                    partition:list,
                    SHUFFLE:bool=True,
                    REMOVE_MEAN:bool=False,
                    data_type:dtype=np.float32,
                    randseed:Optional[int]=None) -> list[np.ndarray]: 
    '''Split the data into sets.
    
    Arguments:\n
        data: a numpy array of data\n
        axis: which axis to split\n
        partition: a list giving the number of snapshots for each set\n
        SHUFFLE: if true the data is shuffled. Default true.\n
        REMOVE_MEAN: if true the data is centered around 0. Default false.\n
        data_type: default numpy.float32.\n
        randseed: an integer as random seed.\n
    Return:\n
        datasets: [data_1, data_2 ...]\n
        means: [mean_of_data_1, mean_of_data_2 ...], empty list if REMOVE_MEAN is false
    '''
    if np.sum(partition) > data.shape[axis]:
        raise ValueError("Not enough snapshots in the dataset.")
    d = np.copy(data) # shuffle is an in-place operation

    d = np.moveaxis(d,axis,0)
    d = d[:np.sum(partition),...]

    if SHUFFLE:
        rng = np.random.default_rng(randseed)
        idx_shuffle, _ = shuffle_with_idx(d.shape[0],rng)
        d = d[idx_shuffle,...]

    # split into sets
    datasets = []
    means = []
    parts = [0]
    parts.extend(partition) 
    for i in range(1,len(partition)+1):
        idx = np.sum(parts[:i])
        a = d[idx:(idx+parts[i]),...]

        if REMOVE_MEAN: 
            # each set is centred around 0.
            a_m = np.mean(a,axis=0)
            means.append(a_m)
            a = a - a_m
            a = np.moveaxis(a,0,axis)
            datasets.append(a.astype(data_type))
        else:
            a = np.moveaxis(a,0,axis)
            datasets.append(a.astype(data_type))
    
    return datasets,means




def shuffle_with_idx(length:int,rng:np.random.Generator):
    '''Create index for shuffling and unshuffling using a random number generator.\n

    Arguments:\n
        length: integer. Number of items to shuffle. For example, to shuffle a dataset with shape (10,2,5) along the first axis, length should be 10.\n
        rng: a numpy random number generator.

    
    '''
    idx_shuffle = np.arange(length)
    rng.shuffle(idx_shuffle)
    idx_unshuffle = np.argsort(idx_shuffle)
    return idx_shuffle, idx_unshuffle