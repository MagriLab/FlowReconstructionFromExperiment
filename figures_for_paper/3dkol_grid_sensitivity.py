import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
import jax
import yaml
import pandas as pd

import flowrec.data as data_utils
from flowrec import losses
from flowrec.training_and_states import restore_trainingstate, params_merge


def _observe(cfg, pred_train_list, data):
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
    observe_kwargs = {key: value for key, value in data.items() if key not in _keys_to_exclude}
    take_observation, _ = cfg.case.observe(
        cfg.data_config,
        example_pred_snapshot = data['u_train'][0][0,...],
        example_pin_snapshot = data['inn_train'][0][0,...],
        **observe_kwargs
    )
    observed_ref = [take_observation(_u) for _u in data['u_train']]
    observed_pred = [take_observation(_u) for _u in pred_train_list]
    return np.concatenate(observed_ref,0), np.concatenate(observed_pred,0)



def _model_and_weights(result_dir, cfg, datainfo, data=None):
    prep_data, make_model = cfg.case.select_model(datacfg=cfg.data_config, mdlcfg=cfg.model_config, traincfg=cfg.train_config)
    mdl = make_model(cfg.model_config)
    state = restore_trainingstate(result_dir,'state')
    if Path(result_dir, 'frozen_params.npy').exists():
        params_frozen = restore_trainingstate(result_dir, 'frozen_params')
        full_params = params_merge(params_frozen, state.params)
        full_params = params_merge(state.params, params_frozen)
    else:
        full_params = state.params

    if data is not None:
        data = prep_data(data, datainfo)
        return mdl, full_params, data
    return mdl, full_params


def _find_num_planes(name):
    f1 = name.split('-')
    nplanes = f1[1][-1]
    return int(nplanes)

def save_summary(result_dir, repeat_dir, csv_name):

    summary = {
        'name': [],
        'num_planes': [],
        'sensor_loss': [],
        'physics_loss': [],
        'relative_error': [],
        'mse':[]
    }
    with open(Path(result_dir,'config.yml'),'r') as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    cfg.data_config.update({'data_dir':'.'+cfg.data_config.data_dir})
    datacfg = cfg.data_config
    traincfg = cfg.train_config
    if traincfg.load_state:
        traincfg.update({'load_state': '.'+traincfg.load_state})
    else:
        traincfg.update({'load_state': str(result_dir.absolute())})
    data, datainfo = cfg.case.dataloader(datacfg)
    if datacfg.shuffle:
        idx_shuffle, idx_unshuffle = data_utils.shuffle_with_idx(np.sum(datacfg.train_test_split), rng = np.random.default_rng(datacfg.randseed))
    forcing = data['forcing']
    mdl, full_params, data = _model_and_weights(result_dir, cfg, datainfo, data=data)
    inn_train = data['inn_train']
    u_train = data['u_train']
    pred_train = []
    for _inn in inn_train:
        pred_train.append(
            mdl.predict(full_params, _inn)
        )
    pred = np.concatenate(pred_train, axis=0)
    ref = np.concatenate(u_train,0)
    observed_ref, observed_pred = _observe(cfg, pred_train, data)
    # 8 plane losses
    summary['name'].append(result_dir.name)
    summary['num_planes'].append(8)
    summary['sensor_loss'].append(float(
        losses.mse(observed_pred, observed_ref)
    ))
    with jax.default_device(jax.devices('cpu')[0]):
        summary['physics_loss'].append(float(
            losses.momentum_loss(pred, datainfo, forcing=forcing) + losses.divergence(pred[...,:-1], datainfo)
        ))
    summary['relative_error'].append(float(losses.relative_error(pred, ref)))
    summary['mse'].append(float(losses.mse(pred, ref)))

    # other number of planes
    folders = [f for f in repeat_dir.iterdir() if f.is_dir()]
    for d in folders:
        print(d.name)
        nplanes = _find_num_planes(d.name)
        with open(Path(d,'config.yml'),'r') as f:
            cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
        cfg.data_config.update({'data_dir':'.'+cfg.data_config.data_dir})
        mdl, full_params = _model_and_weights(d, cfg, datainfo)
        pred_train = []
        for _inn in inn_train:
            pred_train.append(
                mdl.predict(full_params, _inn)
            )
        pred = np.concatenate(pred_train, axis=0)
        observed_ref, observed_pred = _observe(cfg, pred_train, data)
        summary['name'].append(d.name)
        summary['num_planes'].append(nplanes)
        summary['sensor_loss'].append(float(
            losses.mse(observed_pred, observed_ref)
        )) 
        with jax.default_device(jax.devices('cpu')[0]):
            summary['physics_loss'].append(float(
                losses.momentum_loss(pred, datainfo, forcing=forcing) + losses.divergence(pred[...,:-1], datainfo)
            ))
        summary['relative_error'].append(float(losses.relative_error(pred, ref)))
        summary['mse'].append(float(losses.mse(pred, ref)))
    df = pd.DataFrame(summary)
    df = df.set_index('name')
    df.to_csv(csv_name)


if __name__ == "__main__":

    testmin8_dir_share = Path("../local_results/3dkol/sweep_share_8planes/efficient-sweep-23/")
    testmin8_dir_notshare = Path("../local_results/3dkol/sweep_notshare_8planes/summer-sweep-1/")
    testmin_repeats_share = Path("../local_results/3dkol/repeats_planes_share/")
    testmin_repeats_notshare = Path("../local_results/3dkol/repeats_planes_notshare/")

    save_summary(testmin8_dir_share, testmin_repeats_share, testmin_repeats_share / 'summary_3dkol_find_minimum_planes_share.csv')
    save_summary(testmin8_dir_notshare, testmin_repeats_notshare, testmin_repeats_notshare / 'summary_3dkol_find_minimum_planes_notshare')