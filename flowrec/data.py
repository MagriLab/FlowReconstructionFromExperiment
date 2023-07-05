import numpy as np
import jax
import jax.numpy as jnp
import chex

from scipy import linalg
from typing import Optional, Union, List, Sequence, NamedTuple
from dataclasses import dataclass, field

import logging
logger = logging.getLogger(f'fr.{__name__}')

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


def normalise(*args:Array) -> tuple[list[Array], list]:
    '''Normalise array/arrays to between -1 and 1.
    Returns a list of the normalised arrays and a list of range for those arrays.\n
    '''

    ran = []
    data_new = []

    for data in args:
        logging.debug(f'data has type {type(data)} and shape {data.shape}')
        r = [np.min(data), np.max(data)]
        d = 2 * ( (data - r[0]) / (r[1] - r[0]) ) - 1
        ran.append(np.array(r).astype('float32'))
        data_new.append(d)

    return data_new, ran


def unnormalise(data:Array, data_range:Sequence) -> Array:
    '''Un-normalise data.'''

    chex.assert_axis_dimension(data_range,0,2)

    data_new = ((data_range[1] - data_range[0]) * (data + 1) / 2.) + data_range[0]

    return data_new        


@jax.tree_util.Partial(jax.jit,static_argnames=('axis_data','axis_range'))
def unnormalise_group(
        data:Array, 
        data_range:Array, 
        axis_data:int = 0, 
        axis_range:int = 0) -> Array:
    '''Un-normalise stacked data implemented in jax and is jitable. Equivalent to stacking output of `unnormalise(data[i], data_range[j])`\n
    
    Arguments:\n
        data: an array of data.\n
        data_range: an array of data_range.\n
        axis_data: which axis has the data been stacked.\n
        axis_range: which axis has the range been stacked.\n

    Returns:\n
        unnormalised data with the same shape as input data.
    '''
    
    data = jnp.asarray(data)
    data_range = jnp.asarray(data_range)

    _unnomalise_map = jax.vmap(unnormalise, (axis_data,axis_range), axis_data)

    return _unnomalise_map(data,data_range)



def get_whitenoise_std(snr:Scalar,std_signal:Union[Array,Scalar]) -> Union[Array,Scalar]:
    '''Calculate the standard deviation of Gaussian noise based on the signal-to-noise ratio (dB) and the standard deviation of the signal.\n
    
    Arguments:\n
        snr: Scalar, signal-to-noise ratio in dB.\n
        std_data: A scalar or array of the standard deviation of data.\n
    
    Returns;\n
        Standard deviation of noise in the same shape as the input std_signal.
    '''
    snr_l = 10.**(snr/10.)
    std_n = np.sqrt(std_signal**2/snr_l)
    return std_n


def sensor_placement_qrpivot(basis:Array, n_sensors:int, basis_rank:int, **kwargs):
    '''Use the QR pivoting method by Manohar et al. to obtain locations of sensors from a set of tailored basis.\n
    

    Arguments:\n
        basis: the tailored basis for the data, with shape n-by-r, where n is the dimension of the data and r is the rank of the basis.\n
        n_sensors: how many sensors in the domain.\n
        basis_rank: rank r of the basis. \n
    
    Returns:\n
        Q: an unitary matrix.\n
        R: an upper-triangular matrix.\n
        P: the permutation matrix.\n


    Manohar, K., Brunton, B.W., Kutz, J.N., Brunton, S.L., 2018. Data-driven sparse sensor placement for reconstruction: Demonstrating the benefits of exploiting known patterns. IEEE Control Systems Magazine 38, 63–86. https://doi.org/10.1109/MCS.2018.2810460
    '''

    if n_sensors == basis_rank:
        b = basis
    if n_sensors > basis_rank:
        b = basis @ (basis.T)
    
    _, _, p = linalg.qr(b,pivoting=True,**kwargs) 
    return p[:n_sensors]



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
        