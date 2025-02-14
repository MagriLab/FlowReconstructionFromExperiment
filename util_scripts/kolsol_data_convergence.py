import h5py
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pathlib import Path
from flowrec.physics_and_derivatives import get_tke
from flowrec.data import DataMetadata


def plot_mean(figname, *filenames):
    
    sum_sets = []
    nt_sets = []
    tke_sets = []
    kgrid_magnitude_int = None

    for f in filenames:
        print(f)
        with h5py.File(f,'r') as hf:
            u_p = np.array(hf.get('state'))
            ndim = int(hf.get('ndim')[()])
            dt = float(hf.get('dt')[()])
            re = float(hf.get('re')[()])
        ufluc = u_p-np.mean(u_p,axis=0)
        dx = [2*np.pi/u_p.shape[1]]*ndim
        d = [dt]
        d.extend(dx)
        datainfo = DataMetadata(
            re=re,
            discretisation=d,
            axis_index=list(range(ndim+1)),
            problem_2d= (ndim==2)
        ).to_named_tuple()

        if kgrid_magnitude_int is None:
            fftfreq = []
            dk = 2*np.pi/np.array(dx)
            for i in range(ndim):
                _k = np.fft.fftfreq(ufluc.shape[1:-1][i])*dk[i]
                fftfreq.append(_k)
            
            kgrid = np.meshgrid(*fftfreq, indexing='ij')
            kgrid = np.array(kgrid)
            kgrid_magnitude = np.sqrt(np.einsum('n... -> ...', kgrid**2))
            kgrid_magnitude_int = kgrid_magnitude.astype('int')
        spectrum, kbins = get_tke(ufluc, datainfo,kgrid_magnitude=kgrid_magnitude_int)

        tke_sets.append(spectrum)
        nt_sets.append(u_p.shape[0])
        sum_sets.append(np.sum(u_p,axis=0))
    sum_sets = np.array(sum_sets)
    
    sum_all = np.sum(sum_sets,axis=0)
    nt_all = np.sum(nt_sets)

    mean_all = sum_all/nt_all
    tke_all = np.sum(tke_sets,axis=0)

    # print('here')
    fig,axes = plt.subplots(2,ndim+1,figsize=((ndim+1)*3,5))
    axes = axes.flatten()
    axes[0].set_ylabel('y')
    for i in range(ndim+1):
        axes[i].set_xlabel('x')
        if ndim == 2:
            im = axes[i].imshow(mean_all[...,i].T)
        else:
            im = axes[i].imshow(mean_all[...,10,i].T)
        plt.colorbar(im, ax=axes[i])

    # averge over time ploted for each direction
    if ndim == 3:
        x = np.linspace(0, 2*np.pi, mean_all.shape[0])
        axes[-2].plot(x, np.mean(mean_all[...,0], axis=(0,2)), label='y')
        axes[-2].plot(x, np.mean(mean_all[...,0], axis=(0,1)), label='z')
        axes[-2].plot(x, np.mean(mean_all[...,0], axis=(1,2)), label='x')
        axes[-2].plot(x, np.sin(4*x), label='forcing')
        axes[-2].legend()

    # tke
    axes[-1].loglog(kbins, tke_all)
    axes[-1].set(ylabel='TKE', xlabel='wavenumber')

    # Add filenames to the plot
    filenames_text = '\n'.join([Path(f).name for f in filenames])
    axes[ndim+1].text(0.5, 0.5, f'Files used:\n{filenames_text}', ha='center', fontsize=8, wrap=True)
    axes[ndim+1].axis('off')

    fig.tight_layout()
    fig.savefig(figname)


if __name__ == '__main__':



    plot_mean(
        'Kolsol_converge_skipsets(2).png',
        Path('/storage0/ym917/data/simulations/kolsol/dim3_re34_k32_f4_dt01_grid64_478_t200-250.h5'),
        # Path('/storage0/ym917/data/simulations/kolsol/dim3_re34_k32_f4_dt01_grid64_478_t280-330.h5'), # 4GB per file
        Path('/storage0/ym917/data/simulations/kolsol/dim3_re34_k32_f4_dt01_grid64_478_t360-410.h5'),
        Path('/storage0/ym917/data/simulations/kolsol/dim3_re34_k32_f4_dt01_grid64_1372_t200-250.h5'),
        Path('/storage0/ym917/data/simulations/kolsol/dim3_re34_k32_f4_dt01_grid64_691_t200-250.h5'),
        Path('/storage0/ym917/data/simulations/kolsol/dim3_re34_k32_f4_dt01_grid64_57_t180-230.h5'),
    )