import sys
sys.path.append('..')
import h5py
import jax
import yaml
import warnings
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import flowrec.training_and_states as state_utils
import flowrec.data as data_utils
import flowrec.physics_and_derivatives as derivatives

from argparse import ArgumentParser
from pathlib import Path
from flowrec import losses

standard_data_keys = ['u_train_clean', 'u_val_clean', 'train_minmax', 'val_minmax', 'u_train', 'u_val', 'inn_train', 'inn_val', 'y_train', 'y_val']
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

def batching(nb_batches:int, data:jax.Array):
    '''Split data into nb_batches number of batches along axis 0.'''
    return jnp.array_split(data,nb_batches,axis=0)

def get_summary_onecase(d:Path):

    run_name = str(d.name)

    with open(Path(d,'config.yml'),'r') as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    datacfg = cfg.data_config
    mdlcfg = cfg.model_config
    traincfg = cfg.train_config

    data, datainfo = cfg.case.dataloader(datacfg)
    observe_kwargs = {key: value for key, value in data.items() if key not in _keys_to_exclude}
    take_observation, _ = cfg.case.observe(
        datacfg,
        example_pred_snapshot = data['u_train'][0,...],
        example_pin_snapshot = data['inn_train'][0,...],
        **observe_kwargs
    )
    _, train_minmax = take_observation(data['u_train'], init=True)
    _, val_minmax = take_observation(data['u_val'], init=True)
    observed_train = take_observation(data['u_train'])
    observed_val = take_observation(data['u_val'])
    data.update({
        'y_train':observed_train, # not normalised
        'y_val':observed_val, # not normalised
        'train_minmax':train_minmax,
        'val_minmax':val_minmax 
    })

    rng = jax.random.PRNGKey(traincfg.randseed)
    prep_data, make_model = cfg.case.select_model(datacfg = datacfg, mdlcfg = mdlcfg, traincfg = traincfg)
    data = prep_data(data, datainfo)
    mdl = make_model(mdlcfg)
    state = state_utils.restore_trainingstate(d,'state')
    inn_train_batched = batching(traincfg.nb_batches, data['inn_train'])
    if datacfg.snr:
        yfull_train_clean = data['u_train_clean']
        yfull_val_clean = data['u_val_clean']
    else:
        yfull_train_clean = data['u_train']
        yfull_val_clean = data['u_val']

    pred_train = []
    for inn in inn_train_batched:
        pred_train.append(
            mdl.apply(state.params, rng, inn, TRAINING=False)
        )
    pred_train = np.concatenate(pred_train)
    pred_val = mdl.apply(state.params,rng,data['inn_val'],TRAINING=False)

    if cfg.data_config.normalise:
        pred_train = data_utils.unnormalise_group(pred_train, train_minmax, axis_data=-1, axis_range=0)
        pred_val = data_utils.unnormalise_group(pred_val, val_minmax, axis_data=-1, axis_range=0)

        yfull_train_clean = data_utils.unnormalise_group(yfull_train_clean, train_minmax, axis_data=-1, axis_range=0)    
        yfull_val_clean = data_utils.unnormalise_group(yfull_val_clean, val_minmax, axis_data=-1, axis_range=0)    
    observed_train_pred = take_observation(pred_train)
    observed_val_pred = take_observation(pred_val)
    
    loss_train = np.array([
        losses.relative_error(pred_train, yfull_train_clean),
        losses.divergence(pred_train[...,:-1], datainfo),
        losses.momentum_loss(pred_train, datainfo),
        losses.mse(observed_train_pred, observed_train)
    ])
    loss_val = np.array([
        losses.relative_error(pred_val, yfull_val_clean),
        losses.divergence(pred_val[...,:-1], datainfo),
        losses.momentum_loss(pred_val, datainfo),
        losses.mse(observed_val_pred, observed_val)
    ])
    
    return run_name, loss_train, loss_val


def print_best_runs(result_dir, summary_name, summary_loss_train, summary_loss_val):
    loss_train_total = np.sum(summary_loss_train[:,1:],axis=-1)
    loss_val_total = np.sum(summary_loss_val[:,1:],axis=-1)
    loss_train_physics = np.sum(summary_loss_train[:,1:-1],axis=-1)
    loss_val_physics = np.sum(summary_loss_val[:,1:-1],axis=-1)

    idx_train = np.argsort(loss_train_total)
    idx_val = np.argsort(loss_val_total)

    fpath = Path(result_dir, 'best_runs_sorted.txt') 
    fpath.touch(exist_ok=True)
    with open(fpath, 'a') as f:
        f.write('Best runs sorted by total training error: \n')
        f.write('Total loss, Rel L2, Physics loss, Sensor loss')
        counter = 0
        for i in idx_train:
            f.write('\n')
            f.write(f'{counter+1} {summary_name[i]}: {loss_train_total[i]:.5f}, {summary_loss_train[i,0]:.5f}, {loss_train_physics[i]:.4f}, {summary_loss_train[i,-1]:.5f}')
            counter = counter+1


        f.write('\n')
        f.write('\nBest runs sorted by total training error: \n')
        counter = 0
        for i in idx_val:
            f.write(f'{counter+1} {summary_name[i]}: {loss_val_total[i]:.5f}, {summary_loss_val[i,0]:.5f}, {loss_val_physics[i]:.4f}, {summary_loss_val[i,-1]:.5f}')
            counter = counter+1


def plot_correlation(result_dir, summary_loss_train, summary_loss_val):
    loss_train_total = np.sum(summary_loss_train[:,1:],axis=-1)

    corr_train = np.corrcoef(loss_train_total,summary_loss_train[:,0])
    corr_val = np.corrcoef(loss_train_total,summary_loss_val[:,0])

    fig, ax = plt.subplots(1,1)

    ax.scatter(loss_train_total,summary_loss_train[:,0],color='k',label=f'train-train {corr_train[0,1]:.3f}', marker='+')
    ax.scatter(loss_train_total,summary_loss_val[:,0],color='r',label=f'train-val {corr_val[0,1]:.3f}', marker='*',alpha=0.5)

    ax.set(yscale='log',xscale='log')
    ax.legend(loc='upper left')
    ax.set_xlabel('total training loss')
    ax.set_ylabel('rel_l2 training or validation')

    fig.savefig(Path(result_dir,'total_loss_rel_L2_correlation.png'))



def plot_train_validation_compare(result_dir, summary_loss_train, summary_loss_val):
    l_train = summary_loss_train[:,0]
    l_val = summary_loss_val[:,0]

    i = np.arange(len(l_train))

    fig,ax = plt.subplots(1,1)

    ax.scatter(i,l_train,marker='+',label='train',alpha=0.8)
    ax.scatter(i,l_val,marker='*',label='validation',alpha=0.8)
    
    ax.set(yscale='log', xticks=[], ylabel='Rel L2')
    ax.legend()
    fig.savefig(Path(result_dir, 'train-val.png'))


def save_sweep_summary(result_dir: Path):
    if Path(result_dir,'summary.h5').exists():
        warnings.warn('Summary file already exist, reading from existing summary file.')
        with h5py.File(Path(result_dir,'summary.h5'),'r') as hf:
            summary_name = np.array(hf.get('runs_name')).astype('unicode')
            summary_loss_train = np.array(hf.get('runs_loss_train')) # [rel_l2, div, momemtum, sensors]
            summary_loss_val = np.array(hf.get('runs_loss_val')) # [rel_l2, div, momemtum, sensors]
    
    else:
        folder_list = [_d for _d in result_dir.iterdir() if _d.is_dir()]
        counter = 0
        # for each run in the sweep, get summary
        summary_loss_train = []
        summary_loss_val = []
        summary_name = []

        counter = 0
        for d in folder_list:
            run_name, loss_train, loss_val = get_summary_onecase(d)
            print(counter+1, run_name)
            counter += 1
            summary_name.append(run_name)
            summary_loss_train.append(loss_train)
            summary_loss_val.append(loss_val)
        
        summary_loss_train = np.array(summary_loss_train)
        summary_loss_val = np.array(summary_loss_val)

        with h5py.File(Path(result_dir,'summary.h5'),'w') as hf:
            hf.create_dataset('runs_name',data=list(summary_name),dtype=h5py.string_dtype(encoding='utf-8'))
            hf.create_dataset('runs_loss_train',data=summary_loss_train)
            hf.create_dataset('runs_loss_val',data=summary_loss_val) # The columns are [rel_l2, divergence, momentum, sensor]
        

    print_best_runs(result_dir, summary_name, summary_loss_train, summary_loss_val)
    plot_correlation(result_dir, summary_loss_train, summary_loss_val)
    plot_train_validation_compare(result_dir, summary_loss_train, summary_loss_val)


if __name__ == '__main__':
    parser = ArgumentParser(description='Plots and save summary of sweeps.')
    parser.add_argument('result_dir', help="Directory to sweep results")
    args = parser.parse_args()

    save_sweep_summary(Path(args.result_dir))
