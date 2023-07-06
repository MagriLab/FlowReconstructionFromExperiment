import multiprocessing as mp
import numpy as np
import jax
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from absl import app, flags
from functools import partial
from ml_collections import config_flags
from jax.tree_util import Partial

from flowrec.decomposition import POD
from flowrec.data import sensor_placement_qrpivot
from utils.system import temporary_fix_absl_logging

temporary_fix_absl_logging()
#mp.set_start_method('spawn')
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=3'


FLAGS = flags.FLAGS
_CONFIG = config_flags.DEFINE_config_file('cfg','train_config/config.py','Path to training config file.')
flags.DEFINE_integer('n_sensors',4,'number of sensors')
flags.DEFINE_integer('basis_rank',2,'truncate basis')






def load_data(data_loader, datacfg):
    data, datainfo = data_loader(datacfg)
    t_axis = datainfo.axt
    return data['u_train'], t_axis

        
def basis_pod(data,t_axis):
    pod = POD('float32')
    q, _, grid_shape = pod.prepare_data(data,t_axis=t_axis)
    _,_,_,phi = pod.pod(q,method='classic',restore_shape=False)
    return phi[:,:FLAGS.basis_rank],grid_shape


def get_sensor_index(args,return_dict):
    (i,br,grid_shape) = args
    print(f'Performing QR pivoting on component {i}.')
    
    p = sensor_placement_qrpivot(br,FLAGS.n_sensors,FLAGS.basis_rank)
    idx = np.array(np.unravel_index(p,grid_shape)) # [:,0] is the first sensor

    return_dict[i] = idx



def main(_):

    datacfg = FLAGS.cfg.data_config
    dataloader = FLAGS.cfg.case.dataloader

    x,t_axis = load_data(dataloader,datacfg)
#    x = x[:,:100,:20,:]
    print('finished loading data')
    print(x.shape)
   
    ndim = x.shape[-1]
    
    ## get basis
    basis_pod_partial = Partial(basis_pod,t_axis=t_axis)
    vbasis = jax.vmap(basis_pod_partial,in_axes=ndim)
    print('Starting calcaulating pod modes')
    br,shape = vbasis(x)
    print('Finished calculating POD modes')
    shape = jax.numpy.array(shape)
    br = np.split(br,ndim,axis=0)
    shape = np.split(shape,ndim,axis=1)
    for i in range(ndim):
        br[i] = np.squeeze(br[i])
        shape[i] = np.squeeze(shape[i])

    manager = mp.Manager()
    return_dict = manager.dict()
    pool = mp.Pool(3)
    job = partial(get_sensor_index,return_dict=return_dict)
    print('Start QR pivoting')
    pool.map(job,zip(range(3),br,shape))
    pool.close()
    pool.join()
    print('Sensors found')

    idx_list = list(return_dict.values())
    idx = np.concatenate(idx_list,axis=1)

    plt.figure()
    plt.imshow(br[0][:,0].reshape(shape[0]),'jet')
    plt.scatter(idx[1,:],idx[0,:],marker='x',c='k')
    plt.savefig('test.png')

    print('Optimal sensor locations found, writing to sensor_index.txt')

    np.savetxt('sensor_index.txt', idx.astype('int'), fmt='%i', delimiter=",")


if __name__ == '__main__':
    app.run(main)
