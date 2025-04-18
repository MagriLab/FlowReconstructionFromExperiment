import random
import h5py
import numpy as np
import tqdm
import warnings
from argparse import ArgumentParser
# from sys import getsizeof
from datetime import datetime
from typing import Any
from pathlib import Path
from kolsol.torch.solver import KolSol as torchKolSol
from kolsol.numpy.solver import KolSol as numpyKolSol
import torch

def setup_directory(data_path) -> None:

    """Sets up the relevant simulation directory."""

    if not data_path.suffix == '.h5':
        raise ValueError('setup_directory() :: Must pass .h5 data_path')

    if data_path.exists():
        raise FileExistsError(f'setup_directory() :: {data_path} already exists.')

    data_path.parent.mkdir(parents=True, exist_ok=True)


def write_h5(data: dict[str, Any], data_path) -> None:

    """Writes results dictionary to .h5 file.

    Parameters
    ----------
    data: dict[str, Any]
        Data to write to file.
    """

    with h5py.File(data_path, 'w') as hf:

        for k, v in data.items():
            hf.create_dataset(k, data=v)


def generate_data(args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    # raise NotImplementedError

    """Generate Kolmogorov Flow Data."""

    print('00 :: Initialising Kolmogorov Flow Solver.')


    # setting random seed
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    data_path = Path(args.data_path)
    setup_directory(data_path)

    cds = torchKolSol(nk=args.nk, nf=args.nf, re=args.re, ndim=args.ndim, device=device) # type: ignore
    if args.restart_from is not None:
        print(f'restarting from {args.restart_from}')
        with h5py.File(args.restart_from) as hf:
            field_hat = np.array(hf.get('state_hat')[-1, ..., :-1]) 
            field_hat = torch.from_numpy(field_hat).to(device)
            old_dt = float(hf.get('dt')[()])
        if old_dt != args.dt:
            warnings.warn(f'The provied dt {args.dt} does not match the checkpoint dt {old_dt}')
        # print(f'The checkpoint takes up {getsizeof(field_hat)/1024/1024:.5f} MB.')
    else:
        field_hat = cds.random_field(magnitude=10.0, sigma=1.2)
    
    # only run transient if needed
    if args.time_transient > 0.0:
        transients_arange = np.arange(0.0, args.time_transient, args.dt)
        nt_transients = transients_arange.shape[0]
        # integrate over transients
        msg = '01 :: Integrating over transients.'
        for _ in tqdm.trange(nt_transients, desc=msg):
            field_hat += args.dt * cds.dynamics(field_hat)

    # define time-arrays for simulation run
    t_generate = np.arange(0.0, args.time_simulation, args.dt)
    t_save = np.arange(0.0, args.time_simulation, args.dt*args.save_frequency)
    ntg = t_generate.shape[0]
    nts = t_save.shape[0]

    # setup recording arrays - only need to record fourier field
    grid_repeat = [cds.nk_grid]*args.ndim
    state_hat_arr = np.zeros(shape=(nts, *grid_repeat , args.ndim+1), dtype=np.complex128)


    # integrate over simulation domain
    msg = '02 :: Integrating over simulation domain'
    for t in tqdm.trange(ntg, desc=msg):
        # time integrate
        field_hat += args.dt * cds.dynamics(field_hat)

        if t % args.save_frequency == 0:
            # record metrics
            state_hat_arr[int(t/args.save_frequency), ..., :-1] = field_hat.cpu().numpy()
            state_hat_arr[int(t/args.save_frequency), ..., -1] = cds.pressure(field_hat).cpu().numpy()

    data_dict = {
        're': args.re,
        'dt': args.dt*args.save_frequency, 
        'dt_generate': args.dt,
        'nk': args.nk,
        'nf': args.nf,
        'ndim': args.ndim,
        'time': t_save,
        'random_seed': args.random_seed,
        'state_hat': state_hat_arr,
    }

    print('02 :: Writing results to file.')
    write_h5(data_dict,data_path)

    print('03 :: Simulation Done.')



def fourier_to_physical(args):
    
    fourier_data_path = Path(args.fourier_data_path)
    if not fourier_data_path.exists():
        raise ValueError(f'Data path {fourier_data_path} does not exist.')
    
    print('01 :: Reading raw data.')
    with h5py.File(fourier_data_path,'r') as hf:
        nk = int(hf['nk'][()])
        nf = int(hf['nf'][()])
        re = float(hf['re'][()])
        dt = float(hf['dt'][()])
        ndim = int(hf['ndim'][()])
        state_hat = np.array(hf.get('state_hat'))
    print('   :: Finished loading raw data.')

    physical_data_path = Path(args.physical_data_path)
    setup_directory(physical_data_path)

    msg = f'02 :: Moving data to physical domain with {args.ngrid} grid points.'
    nt = state_hat.shape[0]
    batch = 500
    nb_batch = int(nt/batch)
    if (nt % batch) != 0:
        nb_batch = nb_batch + 1
    ks = numpyKolSol(nk=nk, nf=nf, re=re, ndim=ndim)
    state = []
    for i in tqdm.trange(nb_batch, desc=msg):
        state.append(ks.fourier_to_phys(state_hat[i*batch:(i+1)*batch,...], nref=args.ngrid))
    state = np.concatenate(state, axis=0)


    data_dict = {
        'state': state[::args.save_frequency,...],
        'ndim': ndim,
        'nf': nf,
        're': re,
        'dt': dt*args.save_frequency,
        'ngrid': args.ngrid
    }

    print('03 :: Saving data in physical domain.')
    write_h5(data_dict, physical_data_path)


if __name__ == '__main__':

    start_time = datetime.now()

    parser = ArgumentParser(
        prog = 'Generate Kolmogorov data',
    )
    subparsers = parser.add_subparsers(required=True)

    parser_g = subparsers.add_parser('generate')
    parser_g.add_argument(
        'data_path'
    )
    parser_g.add_argument(
        '--time_simulation', 
        default=120.0, 
        type=float, 
        help='Number of seconds to run simulation for.'
    )
    parser_g.add_argument(
        '--time_transient',
        default=240.0,
        type=float,
        help='Number of seconds to run transient simulation for.'
    )
    parser_g.add_argument(
        '--random_seed',
        help='Random seed to start with.',
        required=True,
        type=int
    )
    parser_g.add_argument(
        '--nk',
        default=32,
        type=int,
        help='Number of wavenumber for Kolsol.'
    )
    parser_g.add_argument(
        '--nf',
        default=4,
        type=int,
        help='Forcing frequency.'
    )
    parser_g.add_argument(
        '--re',
        default=42,
        type=int,
        help='Reynolds number.'
    )
    parser_g.add_argument(
        '--ndim',
        default=2,
        type=int,
        help='Simulation dimension.'
    )
    parser_g.add_argument(
        '--dt',
        default=0.005,
        type=float,
        help='Time step.'
    )
    parser_g.add_argument(
        '--restart_from', 
        default=None,
        type=str,
        help='Restart from a file'
    )
    parser_g.add_argument(
        '--save_frequency',
        default=1,
        type=int,
        help='How often to write output. Default write all snapshots.'
    )
    parser_g.set_defaults(func=generate_data)


    parser_t = subparsers.add_parser('fourier_to_physical')
    parser_t.add_argument(
        'fourier_data_path', 
        help='Path to the data generated using Kolsol.'
    )
    parser_t.add_argument(
        'physical_data_path',
        help='Path to save the physical data.'
    )
    parser_t.add_argument(
        'ngrid',
        type=int,
        help='Number of grid points each side in the physical domain.'
    )
    parser_t.add_argument(
        '--save_frequency',
        default=1,
        type=int,
        help='How often to write output. Default write all snapshots.'
    )
    parser_t.set_defaults(func=fourier_to_physical)


    args = parser.parse_args()
    args.func(args)
	
    end_time = datetime.now()
    print(f'Time taken (hh:mm:ss.ms) {end_time-start_time}')