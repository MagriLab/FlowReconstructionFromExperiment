import sys
sys.path.append('..')
import h5py
import jax
import yaml
import warnings
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp
import flowrec.training_and_states as state_utils
import flowrec.data as data_utils
import flowrec.physics_and_derivatives as derivatives

import flowrec.training_and_states as state_utils
import flowrec.data as data_utils
import flowrec.physics_and_derivatives as derivatives

from argparse import ArgumentParser
from pathlib import Path
from flowrec import losses
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from matplotlib import gridspec
from scipy.interpolate import RBFInterpolator
from flowrec.utils import simulation
from flowrec import losses
from flowrec.utils.py_helper import slice_from_tuple
from flowrec.utils.myplots import truegrey, create_custom_colormap, make_cax
cmap_trafficlight = create_custom_colormap('trafficlight')
from flowrec.utils import my_continuous_cmap, my_discrete_cmap
from flowrec.training_and_states import restore_trainingstate, params_split, params_merge, generate_update_fn, TrainingState
from flowrec.data import unnormalise_group, normalise

standard_data_keys = ['u_train_clean', 'u_val_clean', 'train_minmax', 'val_minmax', 'u_train', 'u_val', 'inn_train', 'inn_val', 'y_train', 'y_val']

def get_summary_onecase(
        results_dir:Path,
        idx_z:int,
):
    with open(Path(results_dir,'config.yml'),'r') as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    cfg.data_config.update({'data_dir':'.'+cfg.data_config.data_dir})
    datacfg = cfg.data_config
    traincfg = cfg.train_config

    data, datainfo = cfg.case.dataloader(datacfg)
    if datacfg.shuffle:
        idx_shuffle, idx_unshuffle = data_utils.shuffle_with_idx(np.sum(datacfg.train_test_split), rng = np.random.default_rng(datacfg.randseed))
    _keys_to_exclude = [
        'u_train_clean',
        'u_val_clean',
        'train_minmax',
        'val_minmax',
        'u_train',
        'u_val',
        'inn_train',
        'inn_val'
    ]

    prep_data, make_model = cfg.case.select_model(datacfg=datacfg, mdlcfg=cfg.model_config, traincfg=traincfg)
    data = prep_data(data, datainfo)
    inn_train = data['inn_train']
    u_train = data['u_train']
    # print(u_train[0].shape)
    mdl = make_model(cfg.model_config)
    state = restore_trainingstate(results_dir,'state')


    if Path(results_dir, 'frozen_params.npy').exists():
        params_frozen = restore_trainingstate(results_dir, 'frozen_params')
        # print('These layers are frozen: ')
        # for l in list(params_frozen):
        #     print("  ",l)
        full_params = params_merge(params_frozen, state.params)
        full_params = params_merge(state.params, params_frozen)
    else:
        full_params = state.params

    pred_train = []
    for _inn in inn_train:
        pred_train.append(
            mdl.predict(full_params, _inn)
        )
    pred_train = np.concatenate(pred_train, axis=0)

    observe_kwargs = {key: value for key, value in data.items() if key not in _keys_to_exclude}
    take_observation, insert_observation = cfg.case.observe(
        datacfg,
        example_pred_snapshot = data['u_train'][0][0,...],
        example_pin_snapshot = data['inn_train'][0][0,...],
        **observe_kwargs
    )
    observed_train = [take_observation(_u) for _u in data['u_train']]
    observed_pred_train = take_observation(pred_train)
    observed_ref_train = jnp.concatenate(observed_train, axis=0)

    ## validation set 
    match datacfg.components:
        case 'all':
            s = np.s_[:,:,:,idx_z,:]
        case 'velocity':
            s = np.s_[:,:,:,idx_z,:-1]
        case _:
            raise ValueError("Cannot run this script for components that's not 'all' or 'velocity'.")

    observed_ref_val = [_u[s] for _u in data['u_train']]
    observed_ref_val = jnp.concatenate(observed_ref_val, axis=0)
    observed_pred_val = pred_train[s]
    observed_clean_train = take_observation(data['u_train_clean'])
    observed_clean_val = data['u_train_clean'][s]

    ## Compute losses

    u_train = jnp.concatenate(u_train, axis=0)
    u_train_clean = data['u_train_clean']
    forcing = data['forcing']
    
    l = {
        'rel-l2-pred': float(losses.relative_error(pred_train, u_train_clean)),
        'mse-pred': float(losses.mse(pred_train, u_train_clean)),
        'sensor-train-pred': float(losses.mse(observed_pred_train, observed_ref_train)),
        'sensor-val-pred': float(losses.mse(observed_pred_val, observed_ref_val)),
        'momentum-pred': float(losses.momentum_loss(pred_train, datainfo, forcing=forcing)),
        'div-pred': float(losses.divergence(pred_train[...,:-1], datainfo)),
        'rel-l2-noisy': float(losses.relative_error(u_train, u_train_clean)),
        'mse-noisy': float(losses.mse(u_train, u_train_clean)),
        'sensor-train-noisy': float(losses.mse(observed_ref_train, observed_clean_train)),
        'sensor-val-noisy': float(losses.mse(observed_ref_val, observed_clean_val)),
    }

    return results_dir.name, l







def main(result_dir:Path, idx_z:int):
    dir_list = [d for d in result_dir.iterdir() if d.is_dir()]
    df = None
    for d in dir_list:
        name, l = get_summary_onecase(d, idx_z)
        if df:
            df = pd.concatenate(df, pd.Dataframe(l, index=[name]))
        else:
            df = pd.DataFrame(l, index=[name])
    
    
        

    



if __name__ == '__main__':
    parser = ArgumentParser(description='Produce sweep summary, using a plane to as validation set.')
    parser.add_argument('result_dir', help="Directory to sweep results")
    parser.add_argument('z', type=int, help='z coordinate of the plane to use as validation set.')
    args = parser.parse_args()

    main(Path(args.result_dir), args.z)

