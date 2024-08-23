import sys
sys.path.append('..')
import numpy as np
import h5py
from matplotlib import pyplot as plt
plt.style.use('./flowrec/utils/a4.mplstyle')
from pathlib import Path
from flowrec.lr_schedule import cyclic_cosine_decay_schedule
from flowrec.utils import my_discrete_cmap as cmap
cmap2 = ['deepskyblue', 'limegreen', 'darkorange']


save_figure = False


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
    fig.savefig('lr_schedule')
else:
    fig.show()


def load_learning_curve(results_dir):
    with h5py.File(Path(results_dir,'results.h5'),'r') as hf:
        loss_train = np.array(hf.get("loss_train"))
        # loss_val = np.array(hf.get("loss_val"))
        loss_div = np.array(hf.get("loss_div"))
        loss_momentum = np.array(hf.get("loss_momentum"))
        loss_sensors = np.array(hf.get("loss_sensors"))

    return loss_train, loss_momentum+loss_div, loss_sensors


########### learning curves ################
run_2dtriangle_clean = Path('./local_results/2dtriangle/repeat_clean/clean-45')
ltrain_triangle_clean, lp_triangle_clean, ls_triangle_clean = load_learning_curve(run_2dtriangle_clean)

fig, axes = plt.subplots(2,2,figsize=(7,4),gridspec_kw={'top':0.85, 'left':0.1, 'right': 0.97, 'hspace':0.4})
axes = axes.flatten()
axes[0].semilogy(np.arange(len(lp_triangle_clean))[::100],ltrain_triangle_clean[::100], color=cmap(0), linewidth=1)
# axes[0].semilogy(np.arange(len(lp_triangle_clean))[::100],(lp_triangle_clean+ls_triangle_clean)[::100], color=cmap2[0], linewidth=1, linestyle='--')

run_2dtriangle_noisy = Path('./local_results/2dtriangle/repeat_noisy/noisy_random')
seed_triangle_noisy = '325'
loss_labels = ['$\mathcal{L}^c$','$\mathcal{L}^s$','$\mathcal{L}^m$',]
for i, _snr in enumerate([20,10,5]):
    for j, _lossfn in enumerate(['classic', '3', 'mean3']):
        ltrain ,lp ,ls = load_learning_curve(Path(run_2dtriangle_noisy,f'snr{_snr}_{_lossfn}/{_lossfn}-{seed_triangle_noisy}'))
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
    fig.savefig('learning_curve_2dtriangle')
else:
    plt.show()


run_2dkol_clean = Path('./local_results/2dkol/repeat_clean_minimum/3-224-18')
ltrain_kol_clean, lp_kol_clean, ls_kol_clean = load_learning_curve(run_2dkol_clean)
run_2dkol_clean10 = Path('./local_results/2dkol/repeat_clean_minimum/extreme_case_testruns/k2rpb2pi3240628182435')
ltrain_kol_clean10, lp_kol_clean10, ls_kol_clean10 = load_learning_curve(run_2dkol_clean10)

fig, axes = plt.subplots(2,2,figsize=(7,4),gridspec_kw={'top':0.85, 'left':0.1, 'right': 0.97, 'hspace':0.4})
axes = axes.flatten()
axes[0].semilogy(np.arange(len(lp_kol_clean))[::100],ltrain_kol_clean[::100], color=cmap(1), linewidth=1)
# axes[0].semilogy(np.arange(len(lp_kol_clean))[::100],(lp_kol_clean+ls_kol_clean)[::100], color=cmap2[0], linewidth=1, linestyle='--')
axes[0].semilogy(np.arange(len(lp_kol_clean10))[::100],ltrain_kol_clean10[::100], color=cmap(3), linewidth=1, label='$\mathcal{L}^s$ 10 sensors')
# axes[0].semilogy(np.arange(len(lp_kol_clean))[::100],(lp_kol_clean+ls_kol_clean)[::100], color=cmap2[3], linewidth=1, linestyle='--')
_h, _l = axes[0].get_legend_handles_labels()
run_2dkol_noisy = Path('./local_results/2dkol/repeat_noisy/')
seed_kol_noisy = '7521-42135'
loss_labels = ['$\mathcal{L}^c$','$\mathcal{L}^s$','$\mathcal{L}^m$',]
for i, _snr in enumerate([20,10,5]):
    for j, _lossfn in enumerate(['classic', 'loss3', 'mean3']):
        ltrain ,lp ,ls = load_learning_curve(Path(run_2dkol_noisy,f'snr{_snr}_{_lossfn}-{seed_kol_noisy}'))
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
    fig.savefig('learning_curve_2dkol')
else:
    plt.show()

