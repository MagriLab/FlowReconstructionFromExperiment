import numpy as np
from typing import Optional, Union, List, Sequence, NamedTuple
import jax
from dataclasses import dataclass, field

dtype = Union[str,np.dtype]
Scalar = Union[int, float]
Array = Union[np.ndarray, jax.Array]

def data_partition(data:np.ndarray,
                    axis:int,
                    partition:List,
                    SHUFFLE:bool=True,
                    REMOVE_MEAN:bool=False,
                    NORMALISE:bool=False,
                    data_type:dtype=np.float32,
                    randseed:Optional[int]=None,) -> List[np.ndarray]: 
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
    data_range = []
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


def normalise(*args:Array) -> tuple(Array, list):
    '''Normalise array/arrays to between -1 and 1.
    Returns the normalised arrays and range for those arrays.\n
    '''

    ran = []
    data_new = []

    for data in args:
        r = [np.min(data), np.max(data)]
        d = 2 * ( (data - r[0]) / (r[1] - r[0]) ) - 1
        ran.append(r)
        data_new.append(d)

    return data_new, ran


def unnormalise(data:Array, data_range:Sequence) -> Array:
    '''Un-normalise data.'''

    if (len(data_range) != 2) or (data_range[1] <= data_range[0]):
        raise ValueError(f'data_range must be given as [min, max], currently received {data_range}.')

    data_new = ((data_range[1] - data_range[0]) * (data + 1) / 2.) + data_range[0]

    return data_new        



class _Metadata2d(NamedTuple):
    re: Scalar
    discretisation: Sequence[Scalar]
    axis_index: Sequence[int]
    axt: int 
    axx: int 
    axy: int 
    dt: Scalar
    dx: Scalar
    dy: Scalar

class _Metadata3d(NamedTuple):
    re: Scalar
    discretisation: Sequence[Scalar]
    axis_index: Sequence[int]
    axt: int 
    axx: int 
    axy: int 
    axz: int 
    dt: Scalar 
    dx: Scalar 
    dy: Scalar 
    dz: Scalar 

@dataclass
class DataMetadata():
    '''A collection of information about the dataset.\n

    Initialise with:\n
        re: a scalar, Reynolds number\n
        discretisation: a sequence of [dt,dx,dy] or [dt,dx,dy,dz]\n
        axis_index: a sequence of integers that specify which axis in the data is for [time, x, y] or [time, x, y, z]\n
        problem_2d: True if data is for a 2D flow.
    '''
    re: Scalar
    discretisation: Sequence[Scalar]
    axis_index: Sequence[int]
    problem_2d: bool = True

    axt: int = field(init=False)
    axx: int = field(init=False)
    axy: int = field(init=False)
    axz: int = field(init=False)
    dt: Scalar = field(init=False)
    dx: Scalar = field(init=False)
    dy: Scalar = field(init=False)
    dz: Scalar = field(init=False)


    def __post_init__(self):
        if not np.isscalar(self.re):
            raise ValueError('Reynolds number must be given as a scalar.')

        if not isinstance(self.problem_2d, bool):
            raise ValueError("Flag 'problem_2d' must be boolean.")

        if self.problem_2d:
            if (len(self.discretisation) != 3) or (len(self.axis_index) != 3):
                raise ValueError('Expected 2D data but received unexpected number of velocity componenets.')
        else:
            if (len(self.discretisation) != 4) or (len(self.axis_index) != 4):
                raise ValueError('Expected 3D data but received unexpected number of velocity componenets.')
            
        self.axt = self.axis_index[0]
        self.axx = self.axis_index[1]
        self.axy = self.axis_index[2]
        self.dt = self.discretisation[0]
        self.dx = self.discretisation[1]
        self.dy = self.discretisation[2]
        if not self.problem_2d:
            self.axz = self.axis_index[3]
            self.dz = self.discretisation[3]

    def to_named_tuple(self):
        if self.problem_2d:
            return _Metadata2d(
                self.re,
                self.discretisation,
                self.axis_index,
                self.axt,
                self.axx,
                self.axy,
                self.dt,
                self.dx,
                self.dy
            )
        else:
            return _Metadata3d(
                self.re,
                self.discretisation,
                self.axis_index,
                self.axt,
                self.axx,
                self.axy,
                self.axz,
                self.dt,
                self.dx,
                self.dy,
                self.dz
            )
        