import numpy as np
from _typing import *
from typing import Sequence, Union, Optional
import logging
logger = logging.getLogger(f'fr.{__name__}')


# calculate POD modes, contains a reconstruction function
# Originally a matlab script
class POD:
    '''Performs POD for a data matrix.

    Initiation this class will calculate POD for the data matrix.
    
    Methods:
        reconstruct: reconstruct the data with the specified number of POD modes.
        get_modes: returns modes and eigenvalues.
        get_time_coefficient: returns the time coefficients
    '''

    def __init__(self) -> None:
        pass


    def pod(
            self,
            q:np.array, 
            grid_shape:Optional[list] = None,
            method:str = 'classic',
            weight:Union[str,np.array] = 'ones',
            restore_shape:bool = False
    ):
        if len(q.shape) != 2:
            raise ValueError('Input q must be of shape [nx,nt]')
        
        (nx,nt) = q.shape

        # set weights
        w = self._set_weight(weight,nx)
        
        # perform pod
        if method == 'classic':
            modes, lam, a, phi = self.classic_pod(q,w,nt)
        elif method == 'snapshot':
            modes, lam, a, phi = self.snapshot_pod(q,w,nt)
        else:
            raise ValueError("Please choose a method between 'classic' and 'snapshot'.")
        
        if restore_shape:
            original_shape = grid_shape.extend([-1])
            modes = modes.reshape(original_shape)
        
        return modes, lam, a, phi # modes, eigenvalues, coeff, eigenvectors

    
    
    @staticmethod
    def prepare_data(x: np.array, t_axis:int):
        '''Prepare the data for decomposition.
        
        Data is reshaped into [nx, nt], mean is removed.

        Returns:
        nx: number of data points
        nt: number of snapshots
        grid_shape: the shape of the input (original data)
        q: fluctuating velocity with shape [nx, nt]
        q_mean: mean velocity with length [nx]
        '''

        q = np.moveaxis(x,t_axis,-1)
        grid_shape = list(q.shape[:-1])
        nt = q.shape[-1]
        q = np.reshape(x,(-1,nt))

        # remove mean
        q_mean = np.mean(q,axis=1)
        for ti in range(0,nt):
            q[:,ti] = q[:,ti] - q_mean;  

        return q, q_mean, grid_shape


    @staticmethod
    def _set_weight(weight,nx):
        if weight == 'ones':
            weights = np.ones((nx,1))
        elif isinstance(weight, np.array):
            weights = weights
        else:
            raise ValueError('Choose weights from available options or provide a numpy array.')
        return weights
    

    @staticmethod
    def classic_pod(q:np.array, w:np.array):
        
        if len(q.shape) != 2:
            raise ValueError('Input is the wrong shape.')

        nt = q.shape[-1]
        c = q @ ((q.T)*(w.T))/(nt-1) # 2-point spatial correlation tesnsor: Q*Q'
        # print('C is Hermitian?',np.allclose(C,np.conj(C.T)))
        lam,phi = np.linalg.eigh(c) # right eigenvectors and eigenvalues
        idx = np.argsort(lam) # sort
        idx = np.flip(idx)
        modes = phi[:,idx]
        lam = lam[idx]
        phi = np.copy(modes) # keep the original eigenvectors
        
        # normalise energy in the weighted inner product
        normQ = (modes.T @ modes*w).real**0.5
        modes = modes@np.diag(1/np.diag(normQ))

        # calculate time coefficients
        a = q.T @ phi
        return modes, lam, a, phi
    
    @staticmethod
    def snapshot_pod(q:np.array, w:np.array):
        '''Calculate POD using the snapshot method.
        
        Suitable for when number of snapshots is smaller than the number of data points.''' 

        if len(q.shape) != 2:
            raise ValueError('Input is the wrong shape.')

        nt = q.shape[-1]
        C = (q.T) @ (q*w)/(nt-1) # 2-point temporal correlation tesnsor: q'*q 
        lam,phi = np.linalg.eigh(C)
        idx = np.argsort(np.abs(lam)) # sort
        idx = np.flip(idx)
        phi = phi[:,idx]
        lam = lam[idx]

        # get spatial POD modes: PSI = Q*Phi
        modes = (q@phi)*(1/(lam**0.5).T)

        # calculate time coefficients
        a = q @ phi
        return modes, lam, a, phi
    

    @staticmethod
    def _restore_shape(modes:np.array, grid_shape:list):
        ''' Reshape self.modes into the shape of the input data.'''
        original_shape = grid_shape.copy()
        original_shape.extend([-1])
        modes = np.reshape(modes,original_shape)
        return modes
    

    def reconstruct(method:str, which_modes:Union[int,list], phi:np.array, a:np.array):

        if len(phi.shape) != 2:
            raise ValueError()

        if isinstance(which_modes,int):
            idx = np.s_[:,0:which_modes]
            logger.debug(f'Modes up to {which_modes} is used.')
        elif isinstance(which_modes,list):
            if len(which_modes) == 2:
                idx = np.s_[:,which_modes[0]:which_modes[1]]
                logger.debug(f"Using mode [{which_modes[0]}:{which_modes[1]}]")
            else:
                idx = np.s_[:,which_modes]
                logger.debug(f"Using modes named by user in argument 'which_modes'.")
        else:
            raise ValueError("Invalid argument: which_modes.")

        if method == 'classic':
            q_add = phi[idx] @ a[idx].T
        elif method == 'snapshot':
            q_add = a[idx] @ phi[idx].T
        rebuildv = np.reshape(Q_mean,(-1,1)) + q_add