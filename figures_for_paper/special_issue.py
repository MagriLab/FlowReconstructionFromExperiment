import sys
sys.path.append('..')
import re
import h5py
import yaml
import jax
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from mpl_toolkits.axes_grid1 import ImageGrid
from flowrec.utils import my_discrete_cmap
from flowrec.utils.system import set_gpu
from flowrec.training_and_states import restore_trainingstate
from flowrec.physics_and_derivatives import vorticity, dissipation
from flowrec.postprocessing import *
from flowrec import data as data_utils

stylefile = Path('./flowrec/utils/a4.mplstyle')
if not stylefile.exists():
    stylefile = Path('../', stylefile)
plt.style.use(stylefile)
set_gpu(1)
grey = '#808080'
random_repeat = 2

results_dir = Path('./local_results/2dkol/test_extreme/repeats')
if not results_dir.exists():
    results_dir = Path('../', results_dir)
    if not results_dir.exists():
        raise ValueError


# =========== set up shared functions =============
interpolator = Interpolator().kolmogorov2d

def get_summary_onecase(d:Path, predict_only:bool=False):

    print(f"Get summary from {d}")    
    with open(Path(d,'config.yml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    cfg.data_config.update({'data_dir':'.'+cfg.data_config.data_dir})
    datacfg = cfg.data_config
    mdlcfg = cfg.model_config
    traincfg = cfg.train_config

    data, datainfo = cfg.case.dataloader(datacfg)

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
    take_observation, insert_observation = cfg.case.observe(
        datacfg,
        example_pred_snapshot = data['u_train'][0,...],
        example_pin_snapshot = data['inn_train'][0,...],
        **observe_kwargs
    )
    observed_train, train_minmax = take_observation(data['u_train'], init=True)
    observed_val, val_minmax = take_observation(data['u_val'], init=True)
    data.update({
        'y_train':observed_train, # not normalised
        'y_val':observed_val, # not normalised
        'train_minmax':train_minmax,
        'val_minmax':val_minmax 
    })
    prep_data, make_model = cfg.case.select_model(datacfg = datacfg, mdlcfg = mdlcfg, traincfg = traincfg)
    data = prep_data(data, datainfo)
    mdl = make_model(mdlcfg)
    state = restore_trainingstate(d,'state')
    inn_train = data['inn_train']
    if datacfg.snr:
        yfull_train_clean = data['u_train_clean']
    else:
        yfull_train_clean = data['u_train']

    pred_train = []
    _t = 0
    while _t < inn_train.shape[0]:
        if (_t + 500) < inn_train.shape[0]:
            pred_train.append(
                mdl.predict(state.params, inn_train[_t:_t+500,:])
            )
        else:
            pred_train.append(
                mdl.predict(state.params, inn_train[_t:,:])
            )
        _t = _t + 500

    pred_train = np.concatenate(pred_train, axis=0)

    if cfg.data_config.normalise:
        pred_train = data_utils.unnormalise_group(pred_train, train_minmax, axis_data=-1, axis_range=0)

    if predict_only:
        return pred_train
    else:
        sensor_locs = get_sensor_locs_2d(yfull_train_clean[:5,...], take_observation, insert_observation)

        print("    Interpolating from observations")
        u_interp, observed = interpolator(insert_observation, sensor_locs, yfull_train_clean.shape, observed_train)
    
        return (yfull_train_clean, data['u_train'], u_interp, pred_train), datainfo, observed

def is_case(name):
    if re.compile('case1_.').match(name):
        return 1
    elif re.compile('case2_.').match(name):
        return 2
    elif re.compile('case3_.').match(name):
        return 3
    elif re.compile('case4_.').match(name):
        return 4


def read_summary(d:Path):

    with h5py.File(Path(d,'summary.h5')) as hf:
        names = list(np.array(hf.get('runs_name')).astype('unicode'))
        l_train = np.array(hf.get('runs_loss_train'))
    
    lrel = np.zeros((4,5)) # shape(case,repeats)
    lp = np.zeros((4,5)) # shape(case,repeats)
    ltotal = np.zeros((4,5)) # shape(case,repeats)
    names_array = np.empty((4,5),dtype='S10')

    runnames = [run for run in names if Path(d,run).is_dir()]
    ii = np.zeros(shape=(4,),dtype='int')

    for run_no in range(len(runnames)):
        n = int(is_case(runnames[run_no])-1)
        i = int(ii[n])
        names_array[n,i] = runnames[run_no]
        ltotal[n,i] = np.sum(l_train[run_no,1:])
        lrel[n,i] = l_train[run_no,0]
        lp[n,i] = np.sum(l_train[run_no,1:3])
        ii[n] = ii[n]+1
    
    return ltotal, lrel, lp, names_array



# ============ results ==================
lt, lrel, lp, names = read_summary(results_dir)
ltmean = np.mean(lt, axis=1)
lrelmean = np.mean(lrel*100, axis=1)
lpmean = np.mean(lp, axis=1)
ltstd = np.std(lt, axis=1)
lrelstd = np.std(lrel*100, axis=1)
lpstd = np.std(lp, axis=1)

print('case1, case2, case3, case4')
print(f"Total loss {ltmean} +- {ltstd}")
print(f"Relative loss {lrelmean}% +- {lrelstd}%")
print(f"Physics loss {lpmean} +- {lpstd}")

seed = names[0,random_repeat].decode('utf-8').split('_')[-1]
_results1, datainfo12, observed12 = get_summary_onecase(
    Path(results_dir, 'case1_'+seed)
)
_results2 = get_summary_onecase(
    Path(results_dir, 'case2_'+seed),
    predict_only = True
)
_results3, datainfo34, observed34 = get_summary_onecase(
    Path(results_dir, 'case3_'+seed)
)
_results4 = get_summary_onecase(
    Path(results_dir, 'case4_'+seed),
    predict_only = True
)
results12 = list(_results1)
results12.append(_results2) # [ref,noisy,interp,case1,case2]
results34 = list(_results3)
results34.append(_results4) # [ref,noisy,interp,case3,case4]

results12_vort = []
results34_vort = []
di12 = []
di34 = []
dissipation_jit = jax.jit(jax.tree_util.Partial(dissipation,datainfo=datainfo12))
for data in results12:
    results12_vort.append(vorticity(data[...,:-1], datainfo12))
    _d = dissipation_jit(data[...,:-1])
    _dmean = np.mean(_d, axis=[1,2])
    di12.append(_dmean)
for data in results34:
    results34_vort.append(vorticity(data[...,:-1], datainfo34))
    _d = dissipation_jit(data[...,:-1])
    _dmean = np.mean(_d, axis=[1,2])
    di34.append(_dmean)
extreme_threshold = np.mean(di12[0]) + 2*np.std(di12[0])

# =========== plots ============

t_plt = [330, 800, 1380, 1600] # two extreme evenets two normals

fig1 = plt.figure(figsize=(6,4.5))
vmin = np.min(results12_vort[0][t_plt,...])
vmax = np.max(results12_vort[0][t_plt,...])
grid = ImageGrid(fig1, 111, (4,4), cbar_mode='single', share_all=True)
axes = np.array(grid.axes_all).reshape(4,4)
for i,t in enumerate(t_plt):
    # ref
    imref = axes[0,i].imshow(results12_vort[0][t,...], vmax=vmax, vmin=vmin)
    axes[0,i].spy(observed12[0,:,:,0], alpha=0.6)
    axes[1,i].imshow(results12_vort[2][t,...], vmax=vmax, vmin=vmin)
    axes[2,i].imshow(results12_vort[3][t,...], vmax=vmax, vmin=vmin)
    axes[3,i].imshow(results12_vort[4][t,...], vmax=vmax, vmin=vmin)
    axes[3,i].set(xlabel=f'$t={t*datainfo12.dt:.1f}$',xticks=[])
axes[0,0].set(yticks=[], ylabel='Ref.')
axes[1,0].set(yticks=[], ylabel='Interp.')
axes[2,0].set(yticks=[], ylabel='Case 1.')
axes[3,0].set(yticks=[], ylabel='Case 2.')
cbar = grid.cbar_axes[0].colorbar(imref, label='Vorticity')
fig1.savefig('fig_clean_snapshots')


fig2 = plt.figure(figsize=(6,4.5))
grid = ImageGrid(fig2, 111, (4,4), cbar_mode='single', share_all=True)
axes = np.array(grid.axes_all).reshape(4,4)
for i,t in enumerate(t_plt):
    # ref
    imref = axes[0,i].imshow(results34_vort[1][t,...], vmax=vmax, vmin=vmin)
    axes[0,i].spy(observed34[0,:,:,0], alpha=0.6)
    axes[1,i].imshow(results34_vort[2][t,...], vmax=vmax, vmin=vmin)
    axes[2,i].imshow(results34_vort[3][t,...], vmax=vmax, vmin=vmin)
    axes[3,i].imshow(results34_vort[4][t,...], vmax=vmax, vmin=vmin)
    axes[3,i].set(xlabel=f'$t={t*datainfo34.dt:.1f}$',xticks=[])
axes[0,0].set(yticks=[], ylabel='Noisy')
axes[1,0].set(yticks=[], ylabel='Interp.')
axes[2,0].set(yticks=[], ylabel='Case 1.')
axes[3,0].set(yticks=[], ylabel='Case 2.')
cbar = grid.cbar_axes[0].colorbar(imref, label='Vorticity')
fig2.savefig('fig_noisy_snapshots')




realt = np.arange(len(di12[0]))*datainfo12.dt
fig3, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(realt,di12[0], label='Ref.', color=grey)
axes[0].plot(realt,di12[2], label='Interp.', color='b', linestyle=':')
axes[0].plot(realt,di12[3], label='Case 1', color=my_discrete_cmap(0))
axes[0].plot(realt,di12[4], label='Case 2', color=my_discrete_cmap(1))
axes[0].hlines(extreme_threshold, 0, realt[-1], 'r', linestyle='-.')
axes[0].set_label('Average Dissipation')
axes[0].legend()
axes[1].plot(realt,di34[0], label='Ref.', color=grey)
axes[1].plot(realt,di34[2], label='Interp.', color='b', linestyle=':')
axes[1].plot(realt,di34[3], label='Case 3', color=my_discrete_cmap(0))
axes[1].plot(realt,di34[4], label='Case 4', color=my_discrete_cmap(1))
axes[1].hlines(extreme_threshold, 0, realt[-1], 'r', linestyle='-.')
axes[1].set_label('Average Dissipation')
axes[1].legend()
axes[1].set_xlabel('$t$')
fig3.savefig('Dissipation')