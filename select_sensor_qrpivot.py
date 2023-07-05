import multiprocessing as mp
import numpy as np

from absl import app, flags
from itertools import repeat
from ml_collections import config_flags

from flowrec.decomposition import POD
from flowrec.data import sensor_placement_qrpivot
from utils.system import temporary_fix_absl_logging

temporary_fix_absl_logging()


FLAGS = flags.FLAGS
_CONFIG = config_flags.DEFINE_config_file('cfg','train_config/config.py','Path to training config file.')
flags.DEFINE_string('basis','pod','Which method to calculate basis.')
flags.DEFINE_integer('n_sensors',2,'number of sensors')
flags.DEFINE_integer('basis_rank',2,'truncate basis')






def load_data(data_loader, datacfg):
    data, datainfo = data_loader(datacfg)
    t_axis = datainfo.axt
    return data['u_train'], t_axis

        
def basis_pod(data,t_axis):
    pod = POD('float32')
    q, _, grid_shape = pod.prepare_data(data,t_axis=t_axis)
    _ = pod.pod(q[:5,:5],method='classic',restore_shape=False)
    _,_,_,phi = pod.pod(q,method='classic',restore_shape=False)
    return phi[:FLAGS.basis_rank],grid_shape


def get_sensor_index(args):
    (i,x,t_axis,return_dict) = args
    print(f'Performing QR pivoting on component {i}.')

    print(f'Calculating basis ({i})')
    if FLAGS.basis == 'pod':
        basis_r, grid_shape = basis_pod(x[...,i],t_axis)
    else: 
        raise NotImplementedError
    
    print(f'Finding sensor locations ({i})')
    p = sensor_placement_qrpivot(basis_r,FLAGS.n_sensors,FLAGS.basis_rank)
    idx = np.array(np.unravel_index(p,grid_shape)) # [:,0] is the first sensor

    return_dict[i] = idx



def main(_):

    print(f'Taking {FLAGS.n_sensors} sensors using the frist {FLAGS.basis_rank} vectors of {FLAGS.basis} basis.')

    datacfg = FLAGS.cfg.data_config
    dataloader = FLAGS.cfg.case.dataloader

    x,t_axis = load_data(dataloader,datacfg)
   
    ndim = x.shape[-1]
    

    manager = mp.Manager()
    return_dict = manager.dict()
    pool = mp.Pool(ndim)
    pool.map(
        get_sensor_index,
        zip(range(ndim), repeat(x), repeat(t_axis), repeat(return_dict))
    )
    pool.close()
    pool.join()

    idx_list = list(return_dict.values())
    idx = np.concatenate(idx_list,axis=1)
    
    print('Optimal sensor locations found, writing to sensor_index.txt')

    np.savetxt('sensor_index.txt',idx)


if __name__ == '__main__':
    app.run(main)