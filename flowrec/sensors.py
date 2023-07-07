from ._typing import *
from scipy import linalg
from scipy.spatial import distance


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


def keep_one_point_per_cluster(points:Array, threshold:Scalar, metric='euclidean'):
    '''Remove the all but one points in a cluster measured by a radius
    
    Arguments:
        points: a m-by-n matrix of sensor locations. m is the number of sensors and n is the dimension.
        threshold: points within is radius of another point are removed.
        metric: metric to pass to scipy.distance,cdist method. Default Euclidean.
    
    Return:
        a p-by-n matrix where p are the number of sensors. These locations will not be within radius=threshold of each other.
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

