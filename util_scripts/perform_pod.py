import sys
sys.path.append('../')
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from absl import app, flags 
from ml_collections import config_flags

from flowrec.decomposition import POD
from flowrec.utils.system import temporary_fix_absl_logging
from utils import simulation2d
from flowrec.data import data_partition
from flowrec.utils.py_helper import slice_from_tuple



temporary_fix_absl_logging()


FLAGS = flags.FLAGS
_CONFIG = config_flags.DEFINE_config_file('cfg','train_config/config.py','Path to training config file.')
flags.DEFINE_string('method', 'snapshot', 'which method')
flags.DEFINE_bool('as_component', True, 'Decompose each component separately')
flags.DEFINE_bool('training_only', True, 'Decompose each component separately')


def load_data(cfg):
    x_base = 132
    triangle_base_coords = [49,80]
    (ux,uy,pp) = simulation2d.read_data(cfg.data_config.data_dir,x_base)
    x = np.stack([ux,uy,pp],axis=0)
    # remove parts where uz is not zero
    s = slice_from_tuple(cfg.data_config.slice_to_keep)
    x = x[s]


    ## partition data
    [x_train, x_val, x_test], _ = data_partition(x,
        1,
        cfg.data_config.train_test_split,
        REMOVE_MEAN=False,
        randseed=None,
        SHUFFLE=False
    ) # Do not shuffle, do not remove mean for training with physics informed loss

    return x_train, x_val, x_test

def perform_pod(data,t_axis):
    pod = POD('float32')
    q, q_mean, grid_shape = pod.prepare_data(data,t_axis=t_axis)
    modes, lam, a, phi = pod.pod(q,method=FLAGS.method,restore_shape=True,grid_shape=grid_shape)
    return modes, lam, a, phi

def plot(lam):
    if FLAGS.as_component:
        fig,ax = plt.subplots(2,3,figsize=(8,6),sharex=True)
        for i in range(3):
            ax[0,i].bar(range(1,lam.shape[-1]+1),lam[i,:])
            ax[1,i].plot(range(1,lam.shape[-1]+1),100*np.cumsum(lam[i,:])/np.sum(lam[i,:]))

        ax[1,0].set(xlabel='Mode',ylabel='Cumulative energy %',xlim=[0,20])
        ax[0,0].set(ylabel='Energy')
        
        
    else:
        fig,ax = plt.subplots(1,2,figsize=(8,4),sharex=True)
        ax[0].bar(range(1,lam.shape[-1]+1),lam)
        ax[1].plot(range(1,lam.shape[-1]+1),100*np.cumsum(lam)/np.sum(lam))
        ax[0].set(xlabel='Mode',ylabel='Energy',xlim=[0,20])
        ax[1].set_ylabel('Cumulative energy %')


    plt.grid('both')
    fig.tight_layout()
    fig.savefig('eigenvalue plot')


def main(_):
    cfg = FLAGS.cfg
    x_train, x_val, x_test = load_data(cfg)

    if FLAGS.training_only:
        x = x_train
    else:
        x = jnp.concatenate((x_train,x_val,x_test),axis=1)


    if FLAGS.as_component:
        pod_partial = jtu.Partial(perform_pod,t_axis=0)
        vpod = jax.vmap(pod_partial,in_axes=0)
        modes,lam,a,phi = vpod(x) #[3,x,y,m]
    else:
        modes,lam,a,phi = perform_pod(x,t_axis=1)

    plot(lam)
    
        
        


if __name__ == '__main__':
    app.run(main)