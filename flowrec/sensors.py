import itertools as it
import numpy as np
from ._typing import *
from scipy import linalg
from scipy.spatial import distance
from scipy.interpolate import griddata


def sensor_placement_qrpivot(basis:Array, n_sensors:int, basis_rank:int, **kwargs):
    '''Use the QR pivoting method by Manohar et al. to obtain locations of sensors from a set of tailored basis.\n
    

    Arguments
    ------------
        basis: the tailored basis for the data, with shape n-by-r, where n is the dimension of the data and r is the rank of the basis.\n
        n_sensors: how many sensors in the domain.\n
        basis_rank: rank r of the basis. \n
    
    Returns
    ------------
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


def keep_one_point_per_cluster(points:Array, threshold:Scalar, metric='euclidean'):
    '''Remove the all but one points in a cluster measured by a radius
    
    Arguments:
    -------------
        - points: a m-by-n matrix of sensor locations. m is the number of sensors and n is the dimension.
        - threshold: points within is radius of another point are removed.
        metric: metric to pass to scipy.distance,cdist method. Default Euclidean.
    
    Return:
    ---------------
        - a p-by-n matrix where p are the number of sensors. These locations will not be within radius=threshold of each other.
    '''
    dist_matrix = distance.cdist(points, points, metric=metric)  # Compute pairwise distances between points
    np.fill_diagonal(dist_matrix, np.inf)  # Set diagonal elements to infinity to exclude self-distance
    new_points = np.copy(points)
    n_points = points.shape[0]
    i = 0
    while i < n_points:
        valid_indices = np.all(dist_matrix[[i],:] > threshold, axis=0)
        dist_matrix = dist_matrix[:,valid_indices]
        dist_matrix = dist_matrix[valid_indices,:]

        new_points = new_points[valid_indices,:]
        
        n_points = new_points.shape[0]
        i += 1
    return new_points


def make_points_periodic(points:Array, values:Array, side:float=2*np.pi):

    points1 = np.ones_like(points)
    points1[:,0] = points[:,0] - side
    points1[:,1] = points[:,1] + side
    points2 = np.ones_like(points)
    points2[:,0] = points[:,0]
    points2[:,1] = points[:,1] + side
    points3 = np.ones_like(points)
    points3[:,0] = points[:,0] + side
    points3[:,1] = points[:,1] + side
    points4 = np.ones_like(points)
    points4[:,0] = points[:,0] - side
    points4[:,1] = points[:,1]
    points6 = np.ones_like(points)
    points6[:,0] = points[:,0] + side
    points6[:,1] = points[:,1]
    points7 = np.ones_like(points)
    points7[:,0] = points[:,0] - side
    points7[:,1] = points[:,1] - side
    points8 = np.ones_like(points)
    points8[:,0] = points[:,0]
    points8[:,1] = points[:,1] - side
    points9 = np.ones_like(points)
    points9[:,0] = points[:,0] + side
    points9[:,1] = points[:,1] - side

    points_periodic = np.concatenate([points1,points2,points3,points4,points,points6,points7,points8,points9],axis=0)
    values_periodic = np.tile(values,9)
    return points_periodic, values_periodic 


def griddata_periodic(points:Array, values:Array, grid, method:str, side:float = 2*np.pi):
    points_periodic, values_periodic = make_points_periodic(points, values, side)
    grid_interp = griddata(points_periodic, values_periodic, grid, method)
    return grid_interp


def random_coords_generator(rng:np.random.Generator, n:int , gridsize: Tuple)->np.ndarray:
    """Random but repeatable coordinate generators.\n
    When given the same rng, return the first 'n' coordinates from the same sequence of randomly shuffled coordinates.
    
    Arguments:
    ---------------
        - rng: numpy random number generator
        - n: int, number of sensors
        - gridsize: tuple of ints, for example (10,15) for a dataset with 10 grid points in x direction and 15 in y direction.
    
    Returns:
    -----------------
        - index of shape [len(gridsize), n], can be unpacked into slice np.s_[*index]
    """
    # create all grid points
    datasize = [range(_s) for _s in gridsize]
    coords = np.array(list(it.product(*datasize)))
    # randomly select num_inputs number of grid points
    _idx = np.arange(len(coords))
    rng.shuffle(_idx)
    idx = coords[_idx[:n],...].T

    return idx # [len(gridsize), n]