import sys
import os
from pathlib import Path
sys.path.append('..')
abspath = Path(__file__).resolve()
dname = abspath.parent
os.chdir(dname)
import yaml
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

mpl_teplate = './flowrec/utils/a4.mplstyle'
if Path(mpl_teplate).exists():
    plt.style.use(mpl_teplate)
else:
    plt.style.use('.' + mpl_teplate)

from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from flowrec.utils import my_discrete_cmap
from flowrec.sensors import griddata_periodic
from flowrec.losses import relative_error, mse
from flowrec.physics_and_derivatives import vorticity, get_tke, dissipation, second_order_structure_longitudinal
from flowrec.utils.simulation import kolsol_forcing_term
grey = '#808080'

import flowrec.training_and_states as state_utils
import flowrec.data as data_utils

import logging
logging.getLogger('fr.train_config.train_options').setLevel(logging.WARNING)


def get_summary_onecase(d, no_interpolate=False):

    with open(Path(d,'config.yml'), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    cfg.data_config.update({'data_dir':'.'+cfg.data_config.data_dir})
    datacfg = cfg.data_config
    mdlcfg = cfg.model_config
    traincfg = cfg.train_config

    print('Loading data')
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
    print('Building observation functions')
    take_observation, insert_observation = cfg.case.observe(
        datacfg,
        example_pred_snapshot = data['u_train'][0][0,...],
        example_pin_snapshot = data['inn_train'][0][0,...],
        **observe_kwargs
    )
    _, train_minmax = take_observation(np.concatenate(data['u_train'],axis=0), init=True)
    _, val_minmax = take_observation(np.concatenate(data['u_val'],axis=0), init=True)
    observed_train = [take_observation(_u) for _u in data['u_train']]
    observed_val = [take_observation(_u) for _u in data['u_val']]
    data.update({
        'y_train':observed_train, # not normalised
        'y_val':observed_val, # not normalised
        'train_minmax':train_minmax,
        'val_minmax':val_minmax 
    })
    print('Building model')
    prep_data, make_model = cfg.case.select_model(datacfg = datacfg, mdlcfg = mdlcfg, traincfg = traincfg)
    data = prep_data(data, datainfo)
    mdl = make_model(mdlcfg)
    state = state_utils.restore_trainingstate(d,'state')
    inn_train = data['inn_train']
    if datacfg.snr:
        yfull_train_clean = data['u_train_clean']
    else:
        yfull_train_clean = data['u_train']

    print('Predicting')
    pred_train = []
    for _inn in inn_train:
        pred_train.append(
            mdl.predict(state.params, _inn)
        )
    pred_train = np.concatenate(pred_train, axis=0)

    if cfg.data_config.normalise:
        pred_train = data_utils.unnormalise_group(pred_train, train_minmax, axis_data=-1, axis_range=0)

    if no_interpolate:
        return (np.concatenate(yfull_train_clean,axis=0), pred_train), datainfo
    else:
        sensor_locs = get_sensor_locs(
            yfull_train_clean[0][:5,...],
            take_observation, 
            insert_observation
        )

        print('Interpolating from observations')
        u_interp, observed = interpolate(
            insert_observation, 
            sensor_locs, 
            np.concatenate(yfull_train_clean,axis=0).shape, 
            np.concatenate(observed_train,axis=0)
        )
    
        # return (clean, noisy, interp, predicted)
        return (np.concatenate(yfull_train_clean,axis=0), np.concatenate(data['u_train'],axis=0), u_interp, pred_train), datainfo, observed


def get_sensor_locs(example_train, take_observation_fn, insert_observation_fn):

    sensors_empty = np.empty_like(example_train[[0],...])
    sensors_empty.fill(np.nan)
    grid_x, grid_y = np.mgrid[0:example_train.shape[1], 0:example_train.shape[2]]

    gridx1 = np.repeat(grid_x[None,:,:,None],3,axis=3)
    gridy1 = np.repeat(grid_y[None,:,:,None],3,axis=3)

    idx_x = take_observation_fn(gridx1)
    idx_y = take_observation_fn(gridy1)

    idx_x = insert_observation_fn(jnp.asarray(sensors_empty),jnp.asarray(idx_x))[0,...]
    sensors_loc_x = []
    for i in range(idx_x.shape[-1]):
        sensors_loc_x.append(idx_x[...,i][~np.isnan(idx_x[...,i])].astype(int))

    idx_y = insert_observation_fn(jnp.asarray(sensors_empty),jnp.asarray(idx_y))[0,...]
    sensors_loc_y = []
    for i in range(idx_y.shape[-1]):
        sensors_loc_y.append(idx_y[...,i][~np.isnan(idx_y[...,i])].astype(int))
    
    return [sensors_loc_x, sensors_loc_y]



def interpolate(insert_observation_fn, sensor_locs, data_shape, observed):

    sensors_loc_x, sensors_loc_y = sensor_locs

    compare_interp = []
    nt = data_shape[0]
    ndim = data_shape[-1]

    side_length = data_shape[1]
    g1,g2 = np.mgrid[-side_length:side_length*2, -side_length:side_length*2]
    
    temp_observed = np.empty(data_shape)
    temp_observed.fill(np.nan) #this is noisy
    temp_observed = insert_observation_fn(jnp.asarray(temp_observed),jnp.asarray(observed)) # observed_test is noisy if

    for i in range(ndim):
        _locs = np.stack((sensors_loc_x[i].flatten(),sensors_loc_y[i].flatten()),axis=1)
        for t in range(nt):
            _interp = griddata_periodic(_locs,temp_observed[t,...,i][~np.isnan(temp_observed[t,...,i])],(g1,g2),'cubic',side_length)
            compare_interp.append(_interp[side_length:2*side_length,side_length:2*side_length])

    compare_interp = np.array(compare_interp)
    if ndim > 1:
        compare_interp = np.stack((compare_interp[:nt,...],compare_interp[nt:2*nt,...],compare_interp[2*nt:3*nt,...]),axis=-1)

    return compare_interp, temp_observed


def plot_data(data, datainfo, figname=None):
    fig = plt.figure(figsize=(7.5,4))
    grid = []
    pleft = 0.07
    pbottom = 0.61
    pwidth = 0.23
    pheight = 0.34
    for i in range(3):
        ax = fig.add_subplot(position=(pleft+i*(0.1+pwidth), pbottom, pwidth, pheight))
        im = ax.imshow(np.mean(data,axis=0)[...,i].T)
        divider = make_axes_locatable(ax)
        ax1 = divider.append_axes("right", size="5%", pad=0.0) 
        cbar = plt.colorbar(im,cax=ax1) 
        ax.set_yticks([0,128])
        ax.set_yticklabels([6.28,0])
        ax.set_xticks([0,128])
        ax.set_xticklabels([0,6.28])
        grid.append(ax)

    grid.append(fig.add_subplot(position=(pleft+0.7*pwidth,pbottom-0.5,pwidth,pheight)))
    grid.append(fig.add_subplot(position=(pleft+2.2*pwidth,pbottom-0.5,pwidth,pheight)))


    grid[0].set_title('$\\bar{u}_1$')
    grid[1].set_title('$\\bar{u}_2$')
    grid[2].set_title('$\\bar{p}$')
    grid[0].set_xlabel('$x_1$',labelpad=1)
    grid[1].set_xlabel('$x_1$',labelpad=1)
    grid[2].set_xlabel('$x_1$',labelpad=1)
    grid[0].set_ylabel('$x_2$',labelpad=0.7)

    # dissipation
    di_train_true = dissipation(data[...,:-1], datainfo) # (t,x,y)
    di_mean_train_true = np.mean(di_train_true,axis=[1,2])
    delay = int(np.ceil(5./datainfo.dt))
    grid[3].plot(di_mean_train_true[:-delay],di_mean_train_true[delay:],linewidth=0.6, alpha=0.8, color=my_discrete_cmap(0))
    grid[3].set_xlabel('t',labelpad=1)
    grid[3].set_ylabel('t+$\\tau$',labelpad=0.7)
    grid[3].set_yticks([0.16,0.31])
    grid[3].set_xticks([0.16,0.31])
    grid[3].set_aspect('equal')
    grid[3].set_title('Global Dissipation',fontsize=11)
    
    # spectrum
    spectrum_clean, kbins = get_tke(data[...,:2], datainfo)
    grid[4].plot(kbins, spectrum_clean, color='k', linewidth=1.5)
    grid[4].set(yscale='log', xscale='log', xlabel='wavenumber', ylabel='TKE')
    k_nyquist = (2*np.pi / np.sqrt(2*(datainfo_extra.dx**2))) / 2.
    grid[4].set_xlim([1,k_nyquist])
    # grid[4].set_aspect('equal')

    if figname:
        fig.savefig(figname)
    else:
        fig.show()




if __name__ == "__main__":

    # ============= plot data ================
    (ref, _), datainfo_extra = get_summary_onecase(
        '../local_results/2dkol/repeat_clean_minimum/extreme_case_testruns/thesis_corrections/k2rpb2pi3260211180746/',
        no_interpolate=True
    ) #[clean,noisy,interp,pred]
    plot_data(ref, datainfo_extra, figname='./thesis/data_2dkol_overall_with_tke')
    

    # ========== edge case, strict periodic boundary condition =======
    ## This case is ran for thesis correction. Strict periodic boundary condition from periodic padding.
    results_extra, datainfo_extra, observed_extra = get_summary_onecase(
        '../local_results/2dkol/repeat_clean_minimum/extreme_case_testruns/thesis_corrections/k2rpb2pi3260211180746/',
    ) #[clean,noisy,interp,pred]

    print(relative_error(results_extra[3],results_extra[0]))
    print(relative_error(results_extra[2],results_extra[0]))

    pred_boundary_only = np.copy(results_extra[3])
    pred_boundary_only[:,1:-1,1:-1,:] = 0.0

    ref_boundary_only = np.copy(results_extra[0])
    ref_boundary_only[:,1:-1,1:-1,:] = 0.0

    print("Relative error of boundary points: ", relative_error(pred_boundary_only, ref_boundary_only))
    print("MSE of boundary points: ", mse(pred_boundary_only, ref_boundary_only))

    # vorticity
    vort_train_clean = vorticity(results_extra[0][...,:-1],datainfo_extra)
    vort_train_pred = vorticity(results_extra[3][...,:-1],datainfo_extra)
    vort_train_interp = vorticity(results_extra[2][...,:-1],datainfo_extra)
    # spectrum
    spectrum_pred_extra, kbins = get_tke(results_extra[3][...,:-1] - jnp.mean(results_extra[3][...,:-1],axis=0,keepdims=True), datainfo_extra)
    spectrum_ref, _ = get_tke(results_extra[0][...,:-1] - jnp.mean(results_extra[0][...,:-1],axis=0,keepdims=True), datainfo_extra)
    spectrum_interp_extra, _= get_tke(results_extra[2][...,:-1] - jnp.mean(results_extra[2][...,:-1],axis=0,keepdims=True), datainfo_extra)

    # structure function
    ## Second structure function
    ufluc = results_extra[-1][...,:-1] - np.mean(results_extra[-1][...,:-1], axis=0, keepdims=True)
    su_all, counts = second_order_structure_longitudinal(ufluc[:,1:-1,1:-1,:])
    nt = su_all.shape[0]
    count = counts[0,:,:]
    su = jnp.sum(su_all, axis=0) / count / nt

    ufluc_true = results_extra[0][...,:-1] - np.mean(results_extra[0][...,:-1], axis=0, keepdims=True)
    su_all_true, _ = second_order_structure_longitudinal(ufluc_true[:,1:-1,1:-1,:])
    su_true = jnp.sum(su_all_true, axis=0) / count / nt

    # move from xy grid to r
    x,y = np.indices((count.shape))
    dx = x * datainfo_extra.dx
    dy = y * datainfo_extra.dy
    r = (dx**2 + dy**2)**0.5
    print(r.max())
    unique_values, unique_count =  np.unique(r, return_counts=True)
    print(len(unique_values))
    s_u = np.zeros(len(unique_values))
    s_u_true = np.zeros(len(unique_values))
    for i in range(len(unique_values)):
        idx = r == unique_values[i]
        if np.count_nonzero(idx) != unique_count[i]:
            print(np.count_nonzero(idx), unique_count[i])
            break
        s_u[i] = np.sum(su[idx]) / unique_count[i]
        s_u_true[i] = np.sum(su_true[idx]) / unique_count[i]

    ## plots
    plt_t_step = 1000

    fig = plt.figure(figsize=(7,2.5))
    grid = ImageGrid(fig,(0.05,0.1,0.45,0.85),(3,4),share_all=True,cbar_location='right',cbar_mode='single')
    ax = fig.add_axes((0.68,0.2,0.3,0.7))

    vmax = np.max(vort_train_clean[0:3*plt_t_step:plt_t_step,...])-1
    vmin = np.min(vort_train_clean[0:3*plt_t_step:plt_t_step,...])+1
    axes = grid.axes_all
    for i in range(3):
        im0 = axes[i].imshow(vort_train_clean[plt_t_step*i,:,:].T,vmin=vmin,vmax=vmax)
        im1 = axes[i+4].imshow(vort_train_pred[plt_t_step*i,:,:].T,vmin=vmin,vmax=vmax)
        cbar = grid.cbar_axes[i].colorbar(im0)
        cbar = grid.cbar_axes[i].colorbar(im0,label='Vorticity')
        im2 = axes[i+8].imshow(vort_train_interp[plt_t_step*i,:,:].T,vmin=vmin,vmax=vmax)
        axes[i+8].set_xlabel(xlabel=f'$t={int(plt_t_step*i*datainfo_extra.dt)}$')
    imm0 = axes[3].imshow(np.mean(vort_train_clean,axis=0).T,vmin=vmin,vmax=vmax)
    imm1 = axes[7].imshow(np.mean(vort_train_pred,axis=0).T,vmin=vmin,vmax=vmax)
    imm2 = axes[11].imshow(np.mean(vort_train_interp,axis=0).T,vmin=vmin,vmax=vmax)
    axes[11].set_xlabel('Mean')


    axes[0].set(yticks=[],ylabel='Ref.')
    axes[0].spy(observed_extra[0,...,-1], color='r', marker='s', markersize=2, alpha=0.6, zorder=2)
    axes[0].spy(observed_extra[0,...,0], color='k', marker='s', markersize=2, zorder=5)
    axes[4].set(yticks=[],ylabel='Reconstructed')
    axes[8].set(yticks=[],ylabel='Interp.')
    for g in axes:
        g.set(xticks=[],xticklabels=[])
        g.tick_params(bottom=False,top=False)

    ax.plot(kbins, spectrum_ref, label='Reference', color=grey, alpha=0.5, linewidth=3)
    ax.plot(kbins, spectrum_interp_extra, label='Interpolated', color=my_discrete_cmap(0))
    ax.plot(kbins, spectrum_pred_extra, label='Reconstructed', color=my_discrete_cmap(1),linestyle='--')
    ax.set(yscale='log', xscale='log', xlabel='wavenumber', ylabel='TKE')
    k_nyquist = (2*np.pi / np.sqrt(2*(datainfo_extra.dx**2))) / 2.
    ax.set_xlim([1,k_nyquist])
    ax.legend()

    plt.savefig('./thesis/2dkol_clean_10sensors_overall_periodic_boundary')

    fig, ax = plt.subplots(1,1,figsize=(4,2.8))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.97)
    ax.scatter(unique_values[1:], s_u_true[1:], label='Reference', color=my_discrete_cmap(0),s=1, alpha=1)
    ax.scatter(unique_values[1:], s_u[1:], label='Reconstructed', color=my_discrete_cmap(1),s=1, alpha=0.6)
    ax.plot(unique_values[1:], unique_values[1:]**2, label='$r^2$', color='k', linestyle='dashed', linewidth=1)
    # ax.plot(unique_values[1:], unique_values[1:]**(2/3), label='$r^{2/3}$', color='k', linewidth=1)
    ax.set(yscale='log', xscale='log', xlabel='$r$', ylabel="$S_{LL}(r)$")

    plt.legend()
    fig.savefig('./thesis/2dkol_clean_10sensors_periodic_2nd_structure')
