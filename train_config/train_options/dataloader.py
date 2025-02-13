from flowrec._typing import *
from ml_collections.config_dict import ConfigDict

from flowrec.utils import simulation
from flowrec.utils.py_helper import slice_from_tuple
from flowrec.data import data_partition ,get_whitenoise_std
from flowrec.sensors import random_coords_generator


import itertools as it
import warnings
import logging
logger = logging.getLogger(f'fr.{__name__}')

from absl import flags
FLAGS = flags.FLAGS


def dataloader_example() -> tuple[dict, ClassDataMetadata]:

    '''# Example dataloader function.\n
    
    Load data and returns a dictionary with the following arrays. Pass 'dataloader@example' to the config to use this dataloader in training. \n

    The function must return a dictionary of datasets and a datainfo. \n

    The dictionary must have the following items.\n
        1. u_train: [t,x,y,i] or [t,x,y,z,i]. i is the number of variables. The last dimension must be ordered u,v,p or u,v,w,p.\n
        2. u_val: same rules as u_train.\n
        3. inn_train: input vector [t,j], j is the number of inputs and t is the number of snapshots.\n
        4. inn_val: same rules as inn_train.\n 
        
        If normalising the components, then the dictionary needs to also contain\n
            1. train_minmax is the shape of [(min, max of u), (min, max of v), ..., (min, max of pressure)]. \n
            2. val_minmax with the same shape as train_minmax. \n

    The second returned variable, datainfo, should be a MetadataTree.\n
    '''
    raise NotImplementedError




def dataloader_2dtriangle(cfg:ConfigDict = None) -> tuple[dict, ClassDataMetadata]:
    '''# Load data base on data_config. 
    For use with the generated 2D wake behind the triangle, any Reynolds number.\n
    
    data:\n
        - u_train: pre-processed generated data, has shape [t,x,y,3].\n
        - u_val: same shape as u_train.\n
        - inn_train: pre-processed pressure measurement at the base of the triangle, has shape [t,len].\n
        - inn_val: same shape as inn_train.\n
        - If normalise, data dict also has train_minmax and val_minmax [3,range], range is [min,max]\n
        - If noisy, data dict also has u_train_clean and u_val_clean.
    
    datainfo: of class _Metadata2d.
    
    '''
    if not cfg:
        cfg = FLAGS.cfg.data_config
    data = {}

    x_base = 132
    triangle_base_coords = [49,80]
    logger.debug(f'The base of the triangle is placed at x={x_base}, between y={triangle_base_coords[0]} and {triangle_base_coords[1]}.')

    (ux,uy,pp) = simulation.read_data_2dtriangle(cfg.data_dir,x_base)
    x = np.stack([ux,uy,pp],axis=0)
    logger.debug(f'Simulated wake has shape {x.shape}.')
    # remove parts where uz is not zero
    s = slice_from_tuple(cfg.slice_to_keep)
    x = x[s]
    logger.debug(f'Out of the entire simulated wake, the useful domain has shape {x.shape}, determined by data_config.slice_to_keep.')

    if not cfg.randseed:
        randseed = np.random.randint(1,10000)
        cfg.update({'randseed':randseed})
        logger.info('Make a new random key for loading data.')
    else:
        randseed = cfg.randseed
    rng = np.random.default_rng(randseed)

    # Add white noise
    if cfg.snr:
        logger.info('Saving clean data for calculating true loss')
        [x_train,x_val,_], _ = data_partition(x,1,cfg.train_test_split,REMOVE_MEAN=cfg.remove_mean,randseed=randseed,SHUFFLE=cfg.shuffle) # Do not shuffle, do not remove mean for training with physics informed loss
        [ux_train,uy_train,pp_train] = np.squeeze(np.split(x_train,3,axis=0))
        [ux_val,uy_val,pp_val] = np.squeeze(np.split(x_val,3,axis=0))
        
        u_train = np.stack((ux_train,uy_train,pp_train),axis=-1)
        u_val = np.stack((ux_val,uy_val,pp_val),axis=-1)

        data.update({
            'u_train_clean': u_train,
            'u_val_clean': u_val
        })

        logger.info('Adding white noise to data.')
        try:
            FLAGS._noisy = True
        except flags._exceptions.UnrecognizedFlagError as e:
            warnings.warn(str(e))
            warnings.warn('Are you calling the dataloader from train.py?')
        std_data = np.std(x,axis=(1,2,3),ddof=1)
        std_n = get_whitenoise_std(cfg.snr,std_data)
        noise_ux = rng.normal(scale=std_n[0],size=x[0,...].shape)
        noise_uy = rng.normal(scale=std_n[1],size=x[1,...].shape)
        noise_pp = rng.normal(scale=std_n[2],size=x[2,...].shape)
        noise = np.stack([noise_ux,noise_uy,noise_pp],axis=0)
        x = x + noise

    else:
        data.update({
            'u_train_clean': None,
            'u_val_clean': None
        })


    ## Pre-process data that will be used for training
    logger.info("Pre-process data that will be used for training")   
    [x_train,x_val,_], _ = data_partition(x,1,cfg.train_test_split,REMOVE_MEAN=cfg.remove_mean,randseed=randseed,SHUFFLE=cfg.shuffle) # Do not shuffle, do not remove mean for training with physics informed loss
    logger.info(f'Remove mean is {cfg.remove_mean}')

    [ux_train,uy_train,pp_train] = np.squeeze(np.split(x_train,3,axis=0))
    [ux_val,uy_val,pp_val] = np.squeeze(np.split(x_val,3,axis=0))
    
    pb_train = simulation.take_measurement_base(pp_train,ly=triangle_base_coords,centrex=0)
    pb_val = simulation.take_measurement_base(pp_val,ly=triangle_base_coords,centrex=0)

    # information about the grid
    datainfo = DataMetadata(
        re = cfg.re,
        discretisation=[cfg.dt,cfg.dx,cfg.dy],
        axis_index=[0,1,2],
        problem_2d=True
    ).to_named_tuple()
    logger.debug(f'datainfo is {datainfo}.')

    pb_train = np.reshape(pb_train,(cfg.train_test_split[0],-1))
    pb_val = np.reshape(pb_val,(cfg.train_test_split[1],-1))
    logger.debug(f'Input of the training set has shape {pb_train.shape}.')

    u_train = np.stack((ux_train,uy_train,pp_train),axis=-1)
    u_val = np.stack((ux_val,uy_val,pp_val),axis=-1)
    logger.debug(f'Shapes of u_train and u_val are {u_train.shape} and {u_val.shape}.')


    data.update({
        'u_train': u_train, # [t,x,y,3]
        'u_val': u_val, # [t,x,y,3]
        'inn_train': pb_train, # [t,len]
        'inn_val': pb_val # [t,len]
    })

    return data, datainfo




def _load_kolsol(cfg:ConfigDict, dim:int) -> tuple[dict, ClassDataMetadata]:
    '''Load KolSol data, use dim=2 dor 2D simulation and dim=3 for 3D simulation.'''

    if cfg.remove_mean:
        logging.error('Removing mean at this step introduces information about the clean flow fields to the training samples.')

    data = {}
    logger.debug(f'Loading data with config file {cfg.to_dict()}')

    x, re, dt = simulation.read_data_kolsol(cfg.data_dir)
    logger.debug(f'The simulated kolmogorov flow has shape {x.shape}')
    if re != cfg.re or dt!= cfg.dt:
        cfg.update({'re':re, 'dt':dt})
        logger.error('The provided Reynolds number or time step does not match the data')
        logger.debug(f'Data has Re={re}, dt={dt}')
        if cfg.re < 0.00001 or cfg.dt < 0.0000001:
            raise ValueError

    if not cfg.randseed:
        randseed = np.random.randint(1,10000)
        cfg.update({'randseed':randseed})
        logger.info('Make a new random key for loading data.')
    else:
        randseed = cfg.randseed
    rng = np.random.default_rng(randseed)


    # set up datainfo
    if dim == 3:
        datainfo = DataMetadata(
            re=cfg.re,
            discretisation=[cfg.dt,cfg.dx,cfg.dy,cfg.dz],
            axis_index=[0,1,2,3],
            problem_2d=False
        ).to_named_tuple()
    elif dim == 2:
        datainfo = DataMetadata(
            re=cfg.re,
            discretisation=[cfg.dt,cfg.dx,cfg.dy],
            axis_index=[0,1,2],
            problem_2d=True
        ).to_named_tuple()
    logger.debug(f'Datainfo is {datainfo}.')



    # Add white noise
    if cfg.snr:
        
        [u_train, u_val, _], _ = data_partition(
            x, 
            axis=0, 
            partition=cfg.train_test_split,
            REMOVE_MEAN=cfg.remove_mean,
            randseed=randseed,
            SHUFFLE=cfg.shuffle
        )

        logger.info('Saving clean data for calculating true loss')
        data.update({
            'u_train_clean': u_train,
            'u_val_clean': u_val
        })

        logger.info('Adding white noise to data.')
        try:
            FLAGS._noisy = True
        except flags._exceptions.UnrecognizedFlagError as e:
            warnings.warn(str(e))
            warnings.warn('Are you calling the dataloader from train.py?')
        std_data = np.std(x,axis=tuple(np.arange(dim+1)),ddof=1)
        std_n = get_whitenoise_std(cfg.snr,std_data)
        noise = rng.normal([0]*len(std_n),std_n,size=x.shape)
        x = x + noise
        
    else:
        data.update({
            'u_train_clean': None,
            'u_val_clean': None
        })


    ## Pre-process data that will be used for training
    logger.info("Pre-process data that will be used for training")   
    [u_train, u_val, _], _ = data_partition(
        x, 
        axis=0, 
        partition=cfg.train_test_split,
        REMOVE_MEAN=cfg.remove_mean,
        randseed=randseed,
        SHUFFLE=cfg.shuffle
    )

    ## get inputs
    logger.info('Generating inputs')
    if sum([hasattr(cfg, 'random_input'),hasattr(cfg,'pressure_inlet_slice')]) > 1:
        raise ValueError("Please only specify one between 'random_input' and 'pressure_inlet_slice")

    if hasattr(cfg, 'random_input'):
        sensor_seed, num_inputs = cfg.random_input
        logger.info(f'{num_inputs} random pressure inputs generated using random key specified by the user.')
        observation_rng  = np.random.default_rng(sensor_seed)
        inn_idx = random_coords_generator(
            observation_rng,
            num_inputs,
            x.shape[1:-1]
        )
        logger.debug(f'Coordinates created n a {x.shape[1:-1]} grid.')
        logger.debug(f'Pressure input at grid points {tuple(inn_idx)}.')
        # slice data
        slice_inn = np.s_[:,*inn_idx,-1]
        inn_train = u_train[slice_inn].reshape((-1,num_inputs))
        inn_val = u_val[slice_inn].reshape((-1,num_inputs))
        data.update({'_slice_inn': slice_inn})

    elif hasattr(cfg, 'pressure_inlet_slice'):
        inn_loc = slice_from_tuple(cfg.pressure_inlet_slice)
        s_pressure = (np.s_[:],) + inn_loc + (np.s_[-1],)
        inn_train = u_train[s_pressure].reshape((cfg.train_test_split[0],-1))
        inn_val = u_val[s_pressure].reshape((cfg.train_test_split[1],-1))

    else:
        logger.critical('Pressure input is not defined in config. Please define inputs.')
        raise NotImplementedError


    data.update({
        'u_train': u_train,
        'u_val': u_val,
        'inn_train': inn_train,
        'inn_val': inn_val,
    })


    return data, datainfo


def dataloader_2dkol(cfg:ConfigDict|None = None) -> tuple[dict,ClassDataMetadata]:
    '''Load 2D Kolmogorov flow generated with KolSol.'''
    if not cfg:
        cfg = FLAGS.cfg.data_config
    if cfg.remove_mean:
        warnings.warn('Method of removing mean from the Kolmogorov data has not been implemented. Ignoring remove_mean in configs.')
    
    data, datainfo = _load_kolsol(cfg,2)


    ngrid = data['u_val'].shape[datainfo.axx]
    f = simulation.kolsol_forcing_term(cfg.forcing_frequency,ngrid,2)
    data.update({'forcing': f})
    return data, datainfo

def dataloader_3dkol(cfg:ConfigDict|None = None) -> tuple[dict,ClassDataMetadata]:
    '''Load 3D Kolmogorov flow generated with KolSol.'''
    if not cfg:
        cfg = FLAGS.cfg.data_config
    if cfg.remove_mean:
        warnings.warn('Method of removing mean from the Kolmogorov data has not been implemented. Ignoring remove_mean in configs.')
    
    data, datainfo = _load_kolsol(cfg,3)


    ngrid = data['u_val'].shape[datainfo.axx]
    f = simulation.kolsol_forcing_term(cfg.forcing_frequency,ngrid,3)
    data.update({'forcing': f})
    
    return data, datainfo


def dataloader_3dvolvo(cfg:ConfigDict|None = None) -> tuple[dict,ClassDataMetadata]:
    if not cfg:
        cfg = FLAGS.cfg.data_config
    if cfg.remove_mean:
        warnings.warn('Method of removing mean from the Kolmogorov data has not been implemented. Ignoring remove_mean in configs.')

    data = {}
    
    x, d, re, rho = simulation.read_data_volvo(cfg.data_dir, nondimensional=True)
    if not cfg.randseed:
        randseed = np.random.randint(1,10000)
        cfg.update({'randseed':randseed})
        logger.info('Make a new random key for loading data.')
    else:
        randseed = cfg.randseed
    rng = np.random.default_rng(randseed)
    cfg.update({'re':re, 'dt':d[0], 'dx':d[1], 'dy':d[2], 'dz':d[3]})
    
    datainfo = DataMetadata(
        re=re,
        discretisation=d,
        axis_index=[0,1,2,3],
        problem_2d=False
    ).to_named_tuple()
    logger.debug(f'Datainfo is {datainfo}.')

    if cfg.snr:
        raise NotImplementedError("Noise is not implemented yet for volvorig.")
    else:
        data.update({
            'u_train_clean': None,
            'u_val_clean': None
        })
    
    logger.info("Pre-process data that will be used for training")   
    [u_train, u_val, _], _ = data_partition(
        x, 
        axis=0, 
        partition=cfg.train_test_split,
        REMOVE_MEAN=cfg.remove_mean,
        randseed=randseed,
        SHUFFLE=cfg.shuffle
    )

    ## get inputs
    inn_loc = slice_from_tuple(cfg.pressure_inlet_slice)
    s_pressure = (np.s_[:],) + inn_loc + (np.s_[-1],)
    logger.info(f'Taking input preessure measurememts at {s_pressure}.')

    inn_train = u_train[s_pressure].reshape((cfg.train_test_split[0],-1))
    inn_val = u_val[s_pressure].reshape((cfg.train_test_split[1],-1)) # [t,len]
    data.update({
        'u_train': u_train, # [t,x,y,4]
        'u_val': u_val, # [t,x,y,4]
        'inn_train': inn_train, 
        'inn_val': inn_val
    })

    logger.debug(f'Shape of the training set:{u_train.shape}, shape of the inputs:{inn_train.shape}')

    return data, datainfo