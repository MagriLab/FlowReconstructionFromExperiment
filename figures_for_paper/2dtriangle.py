import sys
import os
from pathlib import Path
sys.path.append('..')
abspath = Path(__file__).resolve()
dname = abspath.parent
os.chdir(dname)
import h5py
import jax
import yaml
import numpy as np
from flowrec import postprocessing
from flowrec import losses
from flowrec import data as data_utils
from flowrec import training_and_states as state_utils
from flowrec import physics_and_derivatives as derivatives
from flowrec.lr_schedule import cyclic_cosine_decay_schedule
from flowrec.utils import simulation
from flowrec.utils import my_discrete_cmap as cmap
from flowrec.utils.system import set_gpu
from flowrec.utils.py_helper import slice_from_tuple
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
cmap2 = ['deepskyblue', 'limegreen', 'darkorange']


################## change these values #################
save_figure = False
save_to = './figs/'
which_gpu = 0
run_2dtriangle_clean = Path('../local_results/2dtriangle/repeat_clean/clean-45')
run_dir_2dtriangle_noisy = Path('../local_results/2dtriangle/repeat_noisy/noisy_random')
run_2dkol_clean = Path('../local_results/2dkol/repeat_clean_minimum/3-224-18')
run_2dkol_clean10 = Path('../local_results/2dkol/repeat_clean_minimum/extreme_case_testruns/k2rpb2pi3240628182435')
run_dir_2dkol_noisy = Path('../local_results/2dkol/repeat_noisy/')


################# set up ##############################
set_gpu(which_gpu)
stylefile = Path('./flowrec/utils/a4.mplstyle')
if not stylefile.exists():
    stylefile = Path('../', stylefile)
plt.style.use(stylefile)


############## learning rate schedule #########################
steps = np.arange(20000)

laminarkws = {'decay_steps':(800,1000,1200,1500,2000),'alpha':(0.3,0.3,0.38,0.38,0.38),'lr_multiplier':(1.0,1.0,1.0,0.7,0.5),'boundaries':(1000,2200,3600,5500,8000)}
laminar_schedule = cyclic_cosine_decay_schedule(
    1,
    **laminarkws
)
laminarlr = laminar_schedule(steps)

cyclic_decay_default_kws = {
    'decay_steps':(800,1000,1200,1500),
    'alpha':(0.3,0.3,0.38,0.38),
    'lr_multiplier':(1.0,0.75,0.5,0.5),
    'boundaries':(1000,2200,3600,5500)
}
turbulent_schedule = cyclic_cosine_decay_schedule(
    1,
    **cyclic_decay_default_kws
)
turbulentlr = turbulent_schedule(steps)

fig,axes = plt.subplots(1,2,sharey=True,figsize=(7,2))
axes[0].plot(steps,laminarlr)
axes[0].set(
    yticks=[1.0,0.2],
    yticklabels=['$\\alpha$','0.2$\\alpha$'],
    ylabel='Learning rate',
    xlabel='Epochs',
    xlim=[0,10000]
)
axes[1].plot(steps,turbulentlr)
axes[1].set(
    yticks=[1.0,0.2],
    yticklabels=['$\\alpha$','0.2$\\alpha$'],
    # ylabel='Learning rate',
    xlabel='Epochs',
    xlim=[0,10000]
)
fig.tight_layout()
if save_figure:
    fig.savefig(save_to+'lr_schedule')
else:
    fig.show()


###################### learning curves ###############################
def load_learning_curve(results_dir):
    with h5py.File(Path(results_dir,'results.h5'),'r') as hf:
        loss_train = np.array(hf.get("loss_train"))
        # loss_val = np.array(hf.get("loss_val"))
        loss_div = np.array(hf.get("loss_div"))
        loss_momentum = np.array(hf.get("loss_momentum"))
        loss_sensors = np.array(hf.get("loss_sensors"))

    return loss_train, loss_momentum+loss_div, loss_sensors


ltrain_triangle_clean, lp_triangle_clean, ls_triangle_clean = load_learning_curve(run_2dtriangle_clean)

fig, axes = plt.subplots(2,2,figsize=(7,4),gridspec_kw={'top':0.85, 'left':0.1, 'right': 0.97, 'hspace':0.4})
axes = axes.flatten()
axes[0].semilogy(np.arange(len(lp_triangle_clean))[::100],ltrain_triangle_clean[::100], color=cmap(0), linewidth=1)
# axes[0].semilogy(np.arange(len(lp_triangle_clean))[::100],(lp_triangle_clean+ls_triangle_clean)[::100], color=cmap2[0], linewidth=1, linestyle='--')

seed_triangle_noisy = '325'
loss_labels = ['$\mathcal{L}^c$','$\mathcal{L}^s$','$\mathcal{L}^m$',]
for i, _snr in enumerate([20,10,5]):
    for j, _lossfn in enumerate(['classic', '3', 'mean3']):
        ltrain ,lp ,ls = load_learning_curve(Path(run_dir_2dtriangle_noisy,f'snr{_snr}_{_lossfn}/{_lossfn}-{seed_triangle_noisy}'))
        axes[i+1].semilogy(np.arange(len(lp))[::100],ltrain[::100], color=cmap(j), linewidth=1, label=f'{loss_labels[j]} training')
        # axes[i+1].semilogy(np.arange(len(lp))[::100],(lp+ls)[::100], color=cmap2[j], linewidth=1, linestyle='--', label=f'{loss_labels[j]} unweighted')
        axes[i+1].set_title(f'SNR={_snr}',fontsize='smaller')
handles, labels = axes[3].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncols=3, bbox_to_anchor=(0.5,1.0), fontsize='smaller')
# fig.tight_layout()
axes[0].set_ylabel('Loss', fontsize='smaller')
axes[0].set_title('Non-noisy', fontsize='smaller')
axes[2].set_ylabel('Loss', fontsize='smaller')
axes[2].set_xlabel('Epochs', fontsize='smaller')
axes[3].set_xlabel('Epochs', fontsize='smaller')
if save_figure:
    fig.savefig(save_to+'learning_curve_2dtriangle')
else:
    plt.show()


ltrain_kol_clean, lp_kol_clean, ls_kol_clean = load_learning_curve(run_2dkol_clean)
ltrain_kol_clean10, lp_kol_clean10, ls_kol_clean10 = load_learning_curve(run_2dkol_clean10)

fig, axes = plt.subplots(2,2,figsize=(7,4),gridspec_kw={'top':0.85, 'left':0.1, 'right': 0.97, 'hspace':0.4})
axes = axes.flatten()
axes[0].semilogy(np.arange(len(lp_kol_clean))[::100],ltrain_kol_clean[::100], color=cmap(1), linewidth=1)
# axes[0].semilogy(np.arange(len(lp_kol_clean))[::100],(lp_kol_clean+ls_kol_clean)[::100], color=cmap2[0], linewidth=1, linestyle='--')
axes[0].semilogy(np.arange(len(lp_kol_clean10))[::100],ltrain_kol_clean10[::100], color=cmap(3), linewidth=1, label='$\mathcal{L}^s$ 10 sensors')
# axes[0].semilogy(np.arange(len(lp_kol_clean))[::100],(lp_kol_clean+ls_kol_clean)[::100], color=cmap2[3], linewidth=1, linestyle='--')
_h, _l = axes[0].get_legend_handles_labels()
seed_kol_noisy = '7521-42135'
loss_labels = ['$\mathcal{L}^c$','$\mathcal{L}^s$','$\mathcal{L}^m$',]
for i, _snr in enumerate([20,10,5]):
    for j, _lossfn in enumerate(['classic', 'loss3', 'mean3']):
        ltrain ,lp ,ls = load_learning_curve(Path(run_dir_2dkol_noisy,f'snr{_snr}_{_lossfn}-{seed_kol_noisy}'))
        axes[i+1].semilogy(np.arange(len(lp))[::100],ltrain[::100], color=cmap(j), linewidth=1, label=f'{loss_labels[j]} training')
        # axes[i+1].semilogy(np.arange(len(lp))[::100],(lp+ls)[::100], color=cmap2[j], linewidth=1, linestyle='--', label=f'{loss_labels[j]} unweighted')
        axes[i+1].set_title(f'SNR={_snr}',fontsize='smaller')
handles, labels = axes[3].get_legend_handles_labels()
handles.append(_h[-1])
labels.append(_l[-1])
fig.legend(handles, labels, loc='upper center', ncols=4, bbox_to_anchor=(0.5,1.0), fontsize='smaller')
# fig.tight_layout()
axes[0].set_ylabel('Loss', fontsize='smaller')
axes[0].set_title('Non-noisy', fontsize='smaller')
axes[2].set_ylabel('Loss', fontsize='smaller')
axes[2].set_xlabel('Epochs', fontsize='smaller')
axes[3].set_xlabel('Epochs', fontsize='smaller')
if save_figure:
    fig.savefig(save_to+'learning_curve_2dkol')
else:
    plt.show()




##################### 2dtriangle #######################
Interpolator = postprocessing.Interpolator()

def get_single_case_predictions_2dtriangle(results_dir:Path, has_noise:bool, predict_only=False):
    print(f"Starting {results_dir}")
    with open(Path(results_dir,'config.yml'),'r') as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    cfg.data_config.update({'data_dir':'.'+cfg.data_config.data_dir})
    x_base = 132
    triangle_base_coords = [49,80]
    (ux,uy,pp) = simulation.read_data_2dtriangle(cfg.data_config.data_dir,x_base)
    x = np.stack([ux,uy,pp],axis=0)
    # remove parts where uz is not zero
    s = slice_from_tuple(cfg.data_config.slice_to_keep)
    x = x[s]

    # information about the grid
    datainfo = data_utils.DataMetadata(
        re = cfg.data_config.re,
        discretisation=[cfg.data_config.dt,cfg.data_config.dx,cfg.data_config.dy],
        axis_index=[0,1,2],
        problem_2d=True
    ).to_named_tuple()

    rng = np.random.default_rng(cfg.data_config.randseed)
    if cfg.data_config.snr:
        [x_train,x_val,x_test], _ = data_utils.data_partition(x,1,cfg.data_config.train_test_split,REMOVE_MEAN=cfg.data_config.remove_mean,randseed=cfg.data_config.randseed,shuffle=cfg.data_config.shuffle) # Do not shuffle, do not remove mean for training with physics informed loss
        [ux_train,uy_train,pp_train] = np.squeeze(np.split(x_train,3,axis=0))
        # [ux_val,uy_val,pp_val] = np.squeeze(np.split(x_val,3,axis=0))
        # [ux_test,uy_test,pp_test] = np.squeeze(np.split(x_test,3,axis=0))
        u_train = np.stack((ux_train,uy_train,pp_train),axis=-1)
        # u_val = np.stack((ux_val,uy_val,pp_val),axis=-1)
        # u_test = np.stack((ux_test,uy_test,pp_test),axis=-1)

        
        std_data = np.std(x,axis=(1,2,3),ddof=1)
        std_n = data_utils.get_whitenoise_std(cfg.data_config.snr,std_data)
        noise_ux = rng.normal(scale=std_n[0],size=x[0,...].shape)
        noise_uy = rng.normal(scale=std_n[1],size=x[1,...].shape)
        noise_pp = rng.normal(scale=std_n[2],size=x[2,...].shape)
        noise = np.stack([noise_ux,noise_uy,noise_pp],axis=0)
        x = x + noise


    [x_train_n,x_val_n,x_test_n], _ = data_utils.data_partition(
        x,
        1,
        cfg.data_config.train_test_split,
        REMOVE_MEAN=cfg.data_config.remove_mean,
        randseed=cfg.data_config.randseed,
        shuffle=cfg.data_config.shuffle
    ) # Do not shuffle, do not remove mean for training with physics informed loss
    [ux_train_n,uy_train_n,pp_train_n] = np.squeeze(np.split(x_train_n,3,axis=0))
    [ux_val_n,uy_val_n,pp_val_n] = np.squeeze(np.split(x_val_n,3,axis=0))
    [ux_test_n,uy_test_n,pp_test_n] = np.squeeze(np.split(x_test_n,3,axis=0))
    u_train_n = np.stack((ux_train_n,uy_train_n,pp_train_n),axis=-1)
    u_val_n = np.stack((ux_val_n,uy_val_n,pp_val_n),axis=-1)
    u_test_n = np.stack((ux_test_n,uy_test_n,pp_test_n),axis=-1)

    pb_train = simulation.take_measurement_base(pp_train_n,ly=triangle_base_coords,centrex=0).reshape((cfg.data_config.train_test_split[0],-1))
    pb_val = simulation.take_measurement_base(pp_val_n,ly=triangle_base_coords,centrex=0).reshape((cfg.data_config.train_test_split[1],-1))
    pb_test = simulation.take_measurement_base(pp_test_n,ly=triangle_base_coords,centrex=0).reshape((cfg.data_config.train_test_split[2],-1))

    take_observation, insert_observation = cfg.case.observe(cfg.data_config, example_pred_snapshot=u_train_n[0,...],example_pin_snapshot=pb_train[0,...])
    observed_train, train_minmax = take_observation(u_train_n,init=True)
    observed_val, val_minmax = take_observation(u_val_n,init=True)
    observed_test, test_minmax = take_observation(u_test_n,init=True)
    
    state = state_utils.restore_trainingstate(results_dir,'state')
    _, make_model = cfg.case.select_model(datacfg=cfg.data_config, mdlcfg=cfg.model_config, traincfg=cfg.train_config)
    mdl = make_model(cfg.model_config)

    if cfg.data_config.normalise:
        [pb_train, pb_val, pb_test], _ = data_utils.normalise(pb_train, pb_val, pb_test, range=[train_minmax[-1],val_minmax[-1],test_minmax[-1]])

    rng = jax.random.PRNGKey(10)

    pb_train_batch = np.array_split(pb_train,2,0)
    pred_train = []
    for inn in pb_train_batch:
        pred_train.append(mdl.apply(state.params,rng,inn,TRAINING=False))
    pred_train = np.concatenate(pred_train)
    # pred_test = mdl.apply(state.params,rng,pb_test,TRAINING=False)
    if cfg.data_config.normalise:
        pred_train = data_utils.unnormalise_group(pred_train, train_minmax, axis_data=-1, axis_range=0)
        # pred_test = data_utils.unnormalise_group(pred_test, test_minmax, axis_data=-1, axis_range=0)
    
    if predict_only:
        return pred_train 

    u_interp, observed = Interpolator.triangle2d(u_train_n, pb_train, cfg.case.observe, cfg.data_config)
    
    print(f'Finished {results_dir}')

    if has_noise:
        return (u_train ,u_train_n, u_interp, pred_train), datainfo, observed
    else:
        return (u_train_n, u_interp, pred_train), datainfo, observed



def make_image_snapshots_vorticity_2dtriangle(data, datainfo, figname, t1):
    t1 = 100
    ref = data[0]
    interp = data[1]
    pred = data[2]

    data_vort = []
    for _data in data:
        _vort = derivatives.vorticity(_data[...,:2], datainfo)
        data_vort.append(
            _vort
        )

    fig = plt.figure(figsize=(7,2))

    grid_1 = ImageGrid(fig, (0.04,0,0.54,0.45), (1,3),cbar_mode='single', share_all=True)
    grid_2 = ImageGrid(fig, (0.04,0.46,0.54,0.45), (1,3),cbar_mode='single', share_all=True)

    grid_r1 = ImageGrid(fig, (0.64,0,0.36,0.45), (1,2), cbar_mode='single', share_all=True)
    grid_r2 = ImageGrid(fig, (0.64,0.46,0.36,0.45), (1,2), cbar_mode='single', share_all=True)


    axes = grid_2.axes_all
    im_ref = axes[0].imshow(data_vort[0][t1,...].T)
    im_interp = axes[1].imshow(data_vort[1][t1,...].T)
    im_pred = axes[2].imshow(data_vort[2][t1,...].T)
    for im in [im_ref,im_interp,im_pred]:
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_2.cbar_axes[0].colorbar(im_ref)
    grid_2.axes_all[0].set(xticks=[],yticks=[])
    
    axes = grid_1.axes_all
    im_ref = axes[0].imshow(ref[t1,...,2].T)
    im_interp = axes[1].imshow(interp[t1,...,2].T)
    im_pred = axes[2].imshow(pred[t1,...,2].T)
    for im in [im_ref,im_interp,im_pred]:
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_1.cbar_axes[0].colorbar(im_ref)
    grid_1.axes_all[0].set(xticks=[],yticks=[])

    ## error
    imerr_interp = grid_r2.axes_all[0].imshow(
        np.abs(data_vort[0][t1,...]-data_vort[1][t1,...]).T
        )
    imerr_pred = grid_r2.axes_all[1].imshow(
        np.abs(data_vort[0][t1,...]-data_vort[2][t1,...]).T
    )
    for im in [imerr_interp,imerr_pred]:
        im.set_clim(imerr_interp.get_clim()[0],imerr_interp.get_clim()[1])
    grid_r2.cbar_axes[0].colorbar(im_ref)
    grid_r2.axes_all[0].set(xticks=[],yticks=[])
    
    imerr_interp = grid_r1.axes_all[0].imshow(np.abs(ref[t1,...,2]-interp[t1,...,2]).T)
    imerr_pred = grid_r1.axes_all[1].imshow(np.abs(ref[t1,...,2]-pred[t1,...,2]).T)
    for im in [imerr_interp, imerr_pred]:
        im.set_clim(imerr_interp.get_clim()[0],imerr_interp.get_clim()[1])
    grid_r1.cbar_axes[0].colorbar(imerr_interp)
    grid_r1.axes_all[0].set(xticks=[],yticks=[])

    

    fig.text(0,0.22,'$p$')
    fig.text(0,0.70,'$v$')
    # fig.text(0,0.81,'$u_1$')
    fig.text(0.09,0.96,'Reference')
    fig.text(0.25,0.96,'Interpolated')
    fig.text(0.44,0.96,'Reconstructed')
    fig.text(0.75,0.99,'Absolute error')
    fig.text(0.66,0.92,'Interpolated')
    fig.text(0.82,0.92,'Reconstructed')
    if save_figure:
        plt.savefig(save_to+figname,bbox_inches='tight')
    else:
        plt.show()


def make_image_pdf(data,figname):

    ## interpolation
    interp_test_nonan = []
    for i in range(3):
        mask = ~np.isnan(data[1][0,...,i])
        interp_test_nonan.append(data[1][...,i][:,mask])

    ## plot
    fig,axes = plt.subplots(1,3,figsize=(7,2.5))
    for i,var in zip(range(3),['$u_1^\prime$','$u_2^\prime$','$p^\prime$']):
        # true
        counts_true,bins_true = np.histogram(data[0][...,i].flatten()-np.mean(data[0][...,i].flatten()), density=True, bins='auto')
        axes[i].stairs(counts_true,bins_true,label='True',linewidth=3, color='#808080',alpha=0.5)
        # interpolation
        counts,bins = np.histogram(interp_test_nonan[i].flatten()-np.mean(interp_test_nonan[i].flatten()), density=True, bins='auto')
        axes[i].stairs(counts,bins,label='Interp.',color='k',linestyle='--')
        
        # prediction
        counts,bins = np.histogram(data[2][...,i].flatten()-np.mean(data[2][...,i].flatten()), density=True, bins='auto')
        axes[i].stairs(counts,bins,label='Reconstructed',color=cmap(0))
        axes[i].set(xlabel=var)
    axes[0].set_ylabel('Probability density')

    handlers = axes[0].get_legend_handles_labels()
    fig.legend(*handlers,loc='upper center', bbox_to_anchor=(0.5, 1.15),ncols=5)

    if save_figure:
        plt.savefig(save_to+figname,bbox_inches='tight')
    else:
        plt.show()


results, datainfo, observed = get_single_case_predictions_2dtriangle(run_2dtriangle_clean, has_noise=False, predict_only=False)
time = 100
make_image_snapshots_vorticity_2dtriangle(results, datainfo, '2dtriangle_clean_snapshots'+str(time), time) 
make_image_pdf(results,'2dtriangle_clean_pdf')


fig,ax = plt.subplots(1,1, figsize=(3.5,2))
ax.imshow(results[0][time,...,-1].T, alpha=0.3)
ax.spy(observed[0,...,-1].T,color='r',marker='s',markersize=2,alpha=0.6)
ax.spy(observed[0,...,0].T,color='k',marker='s',markersize=2,zorder=5)
ax.set(xticks=[],xlabel='$x_1$',yticks=[],ylabel='$x_2$')
if save_figure:
    fig.savefig(save_to+'2dtriangle_clean_sensor_location')

loss_interp = losses.relative_error(results[1],results[2])*100
loss_physics_interp = losses.divergence(results[1][...,:-1],datainfo) + losses.momentum_loss(results[1],datainfo)
loss_physics_ref = losses.divergence(results[0][...,:-1],datainfo) + losses.momentum_loss(results[0],datainfo)

print('')
print('')
print('Case 1: Clean 2D triangle wake')
print(f'Interpolated relative error(%) from sensors is {loss_interp}')
print(f'Interpolated physics loss from sensors is {loss_physics_interp}')
print(f'Reference data has physics loss {loss_physics_ref}')
print('')
print('')




## noisy
def get_summary_one_case(folder):
    with h5py.File(Path(run_dir_2dtriangle_noisy,folder,'summary.h5')) as hf:
        l_train = np.array(hf.get('runs_loss_train'))
        l_val = np.array(hf.get('runs_loss_val'))
        idx = np.argmin(np.sum(l_train[:,1:],axis=-1))
        best_run = np.array(hf.get('runs_name')).astype('unicode')[idx]
    # best_run_path = Path(result_dir,folder,best_run)
    
    lmean = np.array([np.mean(l_train[:,0]), np.mean(l_val[:,0])]) # mean over the repeats [rel_l2 train, rel_l2 val]
    lstd = np.array([np.std(l_train[:,0]), np.std(l_val[:,0])]) # std over the repeats [rel_l2 train, rel_l2 val]
    lpmean = np.array([
        np.mean(np.sum(l_train[:,1:3],axis=1)),
        np.mean(np.sum(l_val[:,1:3],axis=1))
    ])
    lpstd = np.array([
        np.std(np.sum(l_train[:,1:3],axis=1)),
        np.std(np.sum(l_val[:,1:3],axis=1))
    ])
    print(best_run)
 
    return lmean, lstd, lpmean, lpstd

best_run_seed = str(325)

run_snr20_classic = Path(run_dir_2dtriangle_noisy,'snr20_classic/classic-'+best_run_seed)
run_snr10_classic = Path(run_dir_2dtriangle_noisy,'snr10_classic/classic-'+best_run_seed)
run_snr5_classic = Path(run_dir_2dtriangle_noisy,'snr5_classic/classic-'+best_run_seed)
run_snr20_3 = Path(run_dir_2dtriangle_noisy,'snr20_3/3-'+best_run_seed)
run_snr10_3 = Path(run_dir_2dtriangle_noisy,'snr10_3/3-'+best_run_seed)
run_snr5_3 = Path(run_dir_2dtriangle_noisy,'snr5_3/3-'+best_run_seed)
run_snr20_mean3 = Path(run_dir_2dtriangle_noisy,'snr20_mean3/mean3-'+best_run_seed)
run_snr10_mean3 = Path(run_dir_2dtriangle_noisy,'snr10_mean3/mean3-'+best_run_seed)
run_snr5_mean3 = Path(run_dir_2dtriangle_noisy,'snr5_mean3/mean3-'+best_run_seed)

lmean_snr20_classic, lstd_snr20_classic, lpmean_snr20_classic, lpstd_snr20_classic = get_summary_one_case('snr20_classic')
lmean_snr20_3, lstd_snr20_3, lpmean_snr20_3, lpstd_snr20_3 = get_summary_one_case('snr20_3')
lmean_snr20_mean3, lstd_snr20_mean3, lpmean_snr20_mean3, lpstd_snr20_mean3 = get_summary_one_case('snr20_mean3')

lmean_snr10_classic, lstd_snr10_classic, lpmean_snr10_classic, lpstd_snr10_classic = get_summary_one_case('snr10_classic')
lmean_snr10_3, lstd_snr10_3, lpmean_snr10_3, lpstd_snr10_3 = get_summary_one_case('snr10_3')
lmean_snr10_mean3, lstd_snr10_mean3, lpmean_snr10_mean3, lpstd_snr10_mean3, = get_summary_one_case('snr10_mean3')

lmean_snr5_classic, lstd_snr5_classic, lpmean_snr5_classic, lpstd_snr5_classic = get_summary_one_case('snr5_classic')
lmean_snr5_3, lstd_snr5_3, lpmean_snr5_3, lpstd_snr5_3 = get_summary_one_case('snr5_3')
lmean_snr5_mean3, lstd_snr5_mean3, lpmean_snr5_mean3, lpstd_snr5_mean3 = get_summary_one_case('snr5_mean3')

l_mean_classic = []
l_mean_classic.append(lmean_snr20_classic)
l_mean_classic.append(lmean_snr10_classic)
l_mean_classic.append(lmean_snr5_classic)
l_mean_classic = np.array(l_mean_classic)*100
l_std_classic = []
l_std_classic.append(lstd_snr20_classic)
l_std_classic.append(lstd_snr10_classic)
l_std_classic.append(lstd_snr5_classic)
l_std_classic = np.array(l_std_classic)*100

l_mean_3 = []
l_mean_3.append(lmean_snr20_3)
l_mean_3.append(lmean_snr10_3)
l_mean_3.append(lmean_snr5_3)
l_mean_3 = np.array(l_mean_3)*100
l_std_3 = []
l_std_3.append(lstd_snr20_3)
l_std_3.append(lstd_snr10_3)
l_std_3.append(lstd_snr5_3)
l_std_3 = np.array(l_std_3)*100

l_mean_mean3 = []
l_mean_mean3.append(lmean_snr20_mean3)
l_mean_mean3.append(lmean_snr10_mean3)
l_mean_mean3.append(lmean_snr5_mean3)
l_mean_mean3 = np.array(l_mean_mean3)*100
l_std_mean3 = []
l_std_mean3.append(lstd_snr20_mean3)
l_std_mean3.append(lstd_snr10_mean3)
l_std_mean3.append(lstd_snr5_mean3)
l_std_mean3 = np.array(l_std_mean3)*100

lp_mean_classic = []
lp_mean_classic.append(lpmean_snr20_classic)
lp_mean_classic.append(lpmean_snr10_classic)
lp_mean_classic.append(lpmean_snr5_classic)
lp_mean_classic = np.array(lp_mean_classic)
lp_std_classic = []
lp_std_classic.append(lpstd_snr20_classic)
lp_std_classic.append(lpstd_snr10_classic)
lp_std_classic.append(lpstd_snr5_classic)
lp_std_classic = np.array(lp_std_classic)

lp_mean_3 = []
lp_mean_3.append(lpmean_snr20_3)
lp_mean_3.append(lpmean_snr10_3)
lp_mean_3.append(lpmean_snr5_3)
lp_mean_3 = np.array(lp_mean_3)
lp_std_3 = []
lp_std_3.append(lpstd_snr20_3)
lp_std_3.append(lpstd_snr10_3)
lp_std_3.append(lpstd_snr5_3)
lp_std_3 = np.array(lp_std_3)

lp_mean_mean3 = []
lp_mean_mean3.append(lpmean_snr20_mean3)
lp_mean_mean3.append(lpmean_snr10_mean3)
lp_mean_mean3.append(lpmean_snr5_mean3)
lp_mean_mean3 = np.array(lp_mean_mean3)
lp_std_mean3 = []
lp_std_mean3.append(lpstd_snr20_mean3)
lp_std_mean3.append(lpstd_snr10_mean3)
lp_std_mean3.append(lpstd_snr5_mean3)
lp_std_mean3 = np.array(lp_std_mean3)

lp_ref = 0.043 # this is the data

def make_image_snapshots_vorticity(data, datainfo, figname, t1):
    # data is (ref, noisy, interp, classic, loss3, mean3)

    ref = data[0]

    fig = plt.figure(figsize=(7,2.5))
    
    # grids for mean
    grid_b1 = ImageGrid(fig, (0.08,0.00,0.92,0.22), (1,6),cbar_mode='single', share_all=True)
    grid_b2 = ImageGrid(fig, (0.08,0.23,0.92,0.22), (1,6),cbar_mode='single', share_all=True)
    
    # grids for snapshots
    grid_t1 = ImageGrid(fig, (0.08,0.52,0.92,0.22), (1,6),cbar_mode='single', share_all=True)
    grid_t2 = ImageGrid(fig, (0.08,0.75,0.92,0.22), (1,6),cbar_mode='single', share_all=True)


    data_vort = []
    for _data in data:
        data_vort.append(
            derivatives.vorticity(_data[...,:2], datainfo)
        )
    
    # snapshots
    axes = grid_t2.axes_all
    im_ref = axes[0].imshow(
        data_vort[0][t1,...].T
    )
    for j in range(1,6):
        im = axes[j].imshow(
            data_vort[j][t1,...].T
        )
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_t2.cbar_axes[0].colorbar(im_ref)
    grid_t2.axes_all[0].set(xticks=[],yticks=[])
    axes = grid_t1.axes_all
    im_ref = axes[0].imshow(
        ref[t1,...,2].T
    )
    for j in range(1,6):
        im = axes[j].imshow(
            data[j][t1,...,2].T
        )
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_t1.cbar_axes[0].colorbar(im_ref)
    grid_t1.axes_all[0].set(xticks=[],yticks=[])
    
    
    # mean
    axes = grid_b2.axes_all
    im_ref = axes[0].imshow(
        np.mean(data_vort[0],axis=0).T
    )
    for j in range(1,6):
        im = axes[j].imshow(
            np.mean(data_vort[j],axis=0).T
        )
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_b2.cbar_axes[0].colorbar(im_ref)
    grid_b2.axes_all[0].set(xticks=[],yticks=[])
    axes = grid_b1.axes_all
    im_ref = axes[0].imshow(
        np.mean(ref[...,2],axis=0).T
    )
    for j in range(1,6):
        im = axes[j].imshow(
            np.mean(data[j][...,2],axis=0).T
        )
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_b1.cbar_axes[0].colorbar(im_ref)
    grid_b1.axes_all[0].set(xticks=[],yticks=[])
    
    
    fig.text(0.11,0.98,'Reference')
    fig.text(0.28,0.98,'Noisy')
    fig.text(0.41,0.98,'Interpolated')
    fig.text(0.60,0.98,'$\mathcal{L}^c$')
    fig.text(0.75,0.98,'$\mathcal{L}^s$')
    fig.text(0.9,0.98,'$\mathcal{L}^m$')
    fig.text(0.005,0.20,'Mean',rotation='vertical')
    fig.text(0.05,0.1, '$\overline{p}$')
    fig.text(0.05,0.3, '$\overline{v}$')
    fig.text(0.005,0.70,f't={t1*datainfo.dt:.1f}',rotation='vertical')
    fig.text(0.05,0.6, '$p$')
    fig.text(0.05,0.85, '$v$')

    if save_figure:
        plt.savefig(save_to+figname,bbox_inches='tight')

snr = [20,10,5]
fig, (axl, axr) = plt.subplots(1,2,figsize=(7,2.5), sharex=True)
axl.errorbar(snr,l_mean_classic[:,0],yerr=l_std_classic[:,0],label='$\mathcal{L}^c$ ',marker='x',color=cmap(0),linewidth=2.5)
axl.errorbar(snr,l_mean_3[:,0],yerr=l_std_3[:,0],label='$\mathcal{L}^s$ ',marker='x',color=cmap(1),linewidth=2.5)
axl.errorbar(snr,l_mean_mean3[:,0],yerr=l_std_mean3[:,0],label='$\mathcal{L}^m$ ',marker='x',color=cmap(2),linewidth=2.5)
axl.set_ylabel('$\epsilon (\%)$')
axl.set_xticks([5,10,20])
axl.set_xlabel('SNR')
axr.errorbar(snr,lp_mean_classic[:,0],yerr=lp_std_classic[:,0],marker='x',color=cmap(0),linewidth=2.5)
axr.errorbar(snr,lp_mean_3[:,0],yerr=lp_std_3[:,0], marker='x',color=cmap(1),linewidth=2.5)
axr.errorbar(snr,lp_mean_mean3[:,0],yerr=lp_std_mean3[:,0],marker='x',color=cmap(2),linewidth=2.5)
axr.hlines(lp_ref, xmin=5,xmax=20, colors=['k'], linestyles='dashed',label='reference data')
# axr.legend(ncol=1)
axr.set_ylabel('$\mathcal{L}_p$')
axr.set_xticks([5,10,20])
axr.set_xlabel('SNR')
fig.legend(ncol=4,loc='upper center', bbox_to_anchor=(0.5, 1.05))
if save_figure:
    fig.savefig(save_to+'2dtriangle_noisy_compare_lossfn',bbox_inches='tight')
else:
    plt.show()

def make_image_snapshots_vorticity_trianglenoisy(data, datainfo, figname, t1):
    # data is (ref, noisy, interp, classic, loss3, mean3)

    ref = data[0]

    fig = plt.figure(figsize=(7,2.5))
    
    # grids for mean
    grid_b1 = ImageGrid(fig, (0.08,0.00,0.92,0.22), (1,6),cbar_mode='single', share_all=True)
    grid_b2 = ImageGrid(fig, (0.08,0.23,0.92,0.22), (1,6),cbar_mode='single', share_all=True)
    
    # grids for snapshots
    grid_t1 = ImageGrid(fig, (0.08,0.52,0.92,0.22), (1,6),cbar_mode='single', share_all=True)
    grid_t2 = ImageGrid(fig, (0.08,0.75,0.92,0.22), (1,6),cbar_mode='single', share_all=True)


    data_vort = []
    for _data in data:
        data_vort.append(
            derivatives.vorticity(_data[...,:2], datainfo)
        )
    
    # snapshots
    axes = grid_t2.axes_all
    im_ref = axes[0].imshow(
        data_vort[0][t1,...].T
    )
    for j in range(1,6):
        im = axes[j].imshow(
            data_vort[j][t1,...].T
        )
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_t2.cbar_axes[0].colorbar(im_ref)
    grid_t2.axes_all[0].set(xticks=[],yticks=[])
    axes = grid_t1.axes_all
    im_ref = axes[0].imshow(
        ref[t1,...,2].T
    )
    for j in range(1,6):
        im = axes[j].imshow(
            data[j][t1,...,2].T
        )
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_t1.cbar_axes[0].colorbar(im_ref)
    grid_t1.axes_all[0].set(xticks=[],yticks=[])
    
    
    # mean
    axes = grid_b2.axes_all
    im_ref = axes[0].imshow(
        np.mean(data_vort[0],axis=0).T
    )
    for j in range(1,6):
        im = axes[j].imshow(
            np.mean(data_vort[j],axis=0).T
        )
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_b2.cbar_axes[0].colorbar(im_ref)
    grid_b2.axes_all[0].set(xticks=[],yticks=[])
    axes = grid_b1.axes_all
    im_ref = axes[0].imshow(
        np.mean(ref[...,2],axis=0).T
    )
    for j in range(1,6):
        im = axes[j].imshow(
            np.mean(data[j][...,2],axis=0).T
        )
        im.set_clim(im_ref.get_clim()[0],im_ref.get_clim()[1])
    grid_b1.cbar_axes[0].colorbar(im_ref)
    grid_b1.axes_all[0].set(xticks=[],yticks=[])
    
    
    fig.text(0.11,0.98,'Reference')
    fig.text(0.28,0.98,'Noisy')
    fig.text(0.41,0.98,'Interpolated')
    fig.text(0.60,0.98,'$\mathcal{L}^c$')
    fig.text(0.75,0.98,'$\mathcal{L}^s$')
    fig.text(0.9,0.98,'$\mathcal{L}^m$')
    fig.text(0.005,0.20,'Mean',rotation='vertical')
    fig.text(0.05,0.1, '$\overline{p}$')
    fig.text(0.05,0.3, '$\overline{v}$')
    fig.text(0.005,0.70,f't={t1*datainfo.dt:.1f}',rotation='vertical')
    fig.text(0.05,0.6, '$p$')
    fig.text(0.05,0.85, '$v$')

    if save_figure:
        fig.savefig(save_to+figname,bbox_inches='tight')
    else:
        plt.show()

results_20, datainfo, observed_20 = get_single_case_predictions_2dtriangle(run_snr20_classic,has_noise=True,predict_only=False)
results_20 = list(results_20)
for run in [run_snr20_3, run_snr20_mean3]:
    result = get_single_case_predictions_2dtriangle(run,has_noise=True,predict_only=True)
    results_20.append(result)
make_image_snapshots_vorticity(results_20, datainfo, '2dtriangle_noisy20_snapshots'+str(time), time)
results_10, datainfo, observed_10 = get_single_case_predictions_2dtriangle(run_snr10_classic,has_noise=True)
results_10 = list(results_10)
for run in [run_snr10_3, run_snr10_mean3]:
    result = get_single_case_predictions_2dtriangle(run,has_noise=True,predict_only=True)
    results_10.append(result)
make_image_snapshots_vorticity(results_10, datainfo, '2dtriangle_noisy10_snapshots'+str(time), time)
results_5, datainfo, observed_5 = get_single_case_predictions_2dtriangle(run_snr5_classic,has_noise=True)
results_5 = list(results_5)
for run in [run_snr5_3, run_snr5_mean3]:
    result = get_single_case_predictions_2dtriangle(run, has_noise=True,predict_only=True)
    results_5.append(result)
make_image_snapshots_vorticity(results_5, datainfo, '2dtriangle_noisy5_snapshots'+str(time), time)
fig = plt.figure(figsize=(3.5,2))
plt.imshow(results_20[0][0,...,2].T, alpha=0.3)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.spy(observed_20[0,...,2].T, color='r', marker='s', markersize=2, alpha=0.6)
plt.spy(observed_20[0,...,1].T, color='k', marker='s', markersize=2)
plt.xticks([])
plt.yticks([])
if save_figure:
    fig.savefig(save_to+'2dtriangle_noisy_sensor_location')
else:
    plt.show()
print(' ')
print(' ')
print('Mean and standard deviatin of the physics loss for loss classic, strict and mean at')
print(f'SNR20: {lp_mean_classic[0,0]:.2f}+-{lp_std_classic[0,0]:.4f}, {lp_mean_3[0,0]:.3f}+-{lp_std_3[0,0]:.4f}, {lp_mean_mean3[0,0]:.3f}+-{lp_std_mean3[0,0]:.4f}')
print(f'SNR10: {lp_mean_classic[1,0]:.2f}+-{lp_std_classic[1,0]:.4f}, {lp_mean_3[1,0]:.3f}+-{lp_std_3[1,0]:.4f}, {lp_mean_mean3[1,0]:.3f}+-{lp_std_mean3[1,0]:.4f}')
print(f'SNR5: {lp_mean_classic[2,0]:.2f}+-{lp_std_classic[2,0]:.4f}, {lp_mean_3[2,0]:.3f}+-{lp_std_3[2,0]:.4f}, {lp_mean_mean3[2,0]:.3f}+-{lp_std_mean3[2,0]:.4f}')
print(f'Physics loss of the reference data is {lp_ref:.4f}')
print(' ')
print(' ')
print('Mean and standard deviatin of the rel-L2 for loss classic, strict and mean at')
print(f'SNR20: {l_mean_classic[0,0]:.2f}+-{l_std_classic[0,0]:.4f}, {l_mean_3[0,0]:.3f}+-{l_std_3[0,0]:.4f}, {l_mean_mean3[0,0]:.3f}+-{l_std_mean3[0,0]:.4f}')
print(f'SNR10: {l_mean_classic[1,0]:.2f}+-{l_std_classic[1,0]:.4f}, {l_mean_3[1,0]:.3f}+-{l_std_3[1,0]:.4f}, {l_mean_mean3[1,0]:.3f}+-{l_std_mean3[1,0]:.4f}')
print(f'SNR5: {l_mean_classic[2,0]:.2f}+-{l_std_classic[2,0]:.4f}, {l_mean_3[2,0]:.3f}+-{l_std_3[2,0]:.4f}, {l_mean_mean3[2,0]:.3f}+-{l_std_mean3[2,0]:.4f}')
print(' ')
print(' ')