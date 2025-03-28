from flowrec._typing import *
from ml_collections.config_dict import ConfigDict
from pathlib import Path

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

    x = np.einsum('utxy -> txyu', x)
    # Add white noise
    if cfg.snr:
        logger.info('Saving clean data for calculating true loss')
        try:
            FLAGS._noisy = True
        except flags._exceptions.UnrecognizedFlagError as e:
            warnings.warn(str(e))
            warnings.warn('Are you calling the dataloader from train.py?')

        x, u_train_clean, u_val_clean = _add_whitenoise(x, randseed+1, cfg)

        logger.info('Saving clean data for calculating true loss')
        data.update({
            'u_train_clean': u_train_clean,
            'u_val_clean': u_val_clean
        })

    else:
        data.update({
            'u_train_clean': None,
            'u_val_clean': None
        })


    ## Pre-process data that will be used for training
    logger.info("Pre-process data that will be used for training")   
    [x], _ = data_partition(
        x, 
        axis=0, 
        partition=[cfg.nsample],
        REMOVE_MEAN=cfg.remove_mean,
        randseed=randseed,
        shuffle=cfg.shuffle
    )

    pp = x[...,-1]
    pb= simulation.take_measurement_base(pp,ly=triangle_base_coords,centrex=0)

    # information about the grid
    datainfo = DataMetadata(
        re = cfg.re,
        discretisation=[cfg.dt,cfg.dx,cfg.dy],
        axis_index=[0,1,2],
        problem_2d=True
    ).to_named_tuple()
    logger.debug(f'datainfo is {datainfo}.')
    
    ## batching
    x = batching(cfg.batch_size, x)
    pb = batching(cfg.batch_size, pb)
    u_train, u_val = [], []
    pb_train, pb_val = [], []
    _idx = np.array(list(range(len(x))))
    _idx_val = _idx[list(cfg.val_batch_idx)]
    for i in _idx:
        if i in _idx_val:
            u_val.append(x[i])
            pb_val.append(pb[i])
        else:
            u_train.append(x[i])
            pb_train.append(pb[i])
    logger.debug(f'Input of the training set (first batch) has shape {pb_train[0].shape}.')

    logger.debug(f'Shapes of u_train and u_val (first batch) are {u_train[0].shape} and {u_val[0].shape}.')


    data.update({
        'u_train': u_train, # [t,x,y,3]
        'u_val': u_val, # [t,x,y,3]
        'inn_train': pb_train, # [t,len]
        'inn_val': pb_val # [t,len]
    })

    return data, datainfo




def _load_kolsol(cfg:ConfigDict, dim:int, multiplesets:bool = False) -> tuple[dict, ClassDataMetadata]:
    '''Load KolSol data, use dim=2 dor 2D simulation and dim=3 for 3D simulation.'''

    if cfg.remove_mean:
        logging.error('Removing mean at this step introduces information about the clean flow fields to the training samples.')

    data = {}
    logger.debug(f'Loading data with config file {cfg.to_dict()}')

    if multiplesets:
        logger.info('Loading multiple sets of data.')
        datapath = Path(cfg.data_dir)
        parent_dir = datapath.parent
        with open(datapath) as f:
            datasets_path = [parent_dir/line.rstrip() for line in f]
        for d in datasets_path:
            if not d.exists():
                raise ValueError(f'{d} does not exist.')
        logger.info('Keep the last dataset unseen for testing.')
        datasets_path = datasets_path[:-1] # keep the last set for testing
        _check_re = []
        _check_dt = []
        x = []
        sets_index = []
        i = 0
        while int(np.sum(sets_index)) < cfg.nsample:
            _x, re, dt = simulation.read_data_kolsol(datasets_path[i])
            x.append(_x)
            sets_index.append(_x.shape[0]) # keep the number of snapshots in each set
            _check_dt.append(dt)
            _check_re.append(re)
            # check if the datasets are compatible
            if len(set(_check_dt)) > 1 or len(set(_check_re)) > 1:
                raise ValueError('Datasets must be generated with the same parameters.')
            i += 1
        x = np.concatenate(x, axis=0) # build a large dataset
        sets_index[-1] = sets_index[-1] - (x.shape[0]-cfg.nsample) # modify the last index if not the entire set is used
    else:    
        x, re, dt = simulation.read_data_kolsol(cfg.data_dir)
        logger.debug(f'The simulated kolmogorov flow has shape {x.shape}')
        sets_index = None # only one set
    if re != cfg.re or dt!= cfg.dt:
        cfg.update({'re':re, 'dt':dt})
        logger.error('The provided Reynolds number or time step does not match the data')
        logger.debug(f'Data has Re={re}, dt={dt}')
        if cfg.re < 0.00001 or cfg.dt < 0.0000001:
            raise ValueError
    logger.info('Cropping data as requested.')
    crop_data = ((None,),) + cfg.crop_data + ((None,),) 
    crop_data = slice_from_tuple(crop_data)
    x = x[crop_data]

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
        try:
            FLAGS._noisy = True
        except flags._exceptions.UnrecognizedFlagError as e:
            warnings.warn(str(e))
            warnings.warn('Are you calling the dataloader from train.py?')
         
        x, u_train_clean, u_val_clean = _add_whitenoise(x, randseed+1, cfg, sets_index)

        logger.info('Saving clean data for calculating true loss')
        data.update({
            'u_train_clean': u_train_clean,
            'u_val_clean': u_val_clean
        })
               
    else:
        data.update({
            'u_train_clean': None,
            'u_val_clean': None
        })


    ## Pre-process data that will be used for training
    logger.info("Pre-process data that will be used for training")   
    [u], _ = data_partition(
        x, 
        axis=0, 
        partition=[cfg.nsample],
        REMOVE_MEAN=cfg.remove_mean,
        randseed=randseed,
        shuffle=cfg.shuffle
    )

    ## Batching and take validation set 
    u = batching(cfg.batch_size, u, sets_index)
    u_train, u_val = [], []
    _idx = np.array(list(range(len(u))))
    _idx_val = _idx[list(cfg.val_batch_idx)]
    for i in _idx:
        if i in _idx_val:
            u_val.append(u[i])
        else:
            u_train.append(u[i])
    logger.debug(f"{np.sum([_u.shape[0] for _u in u_val])} snapshots are used for validation.")

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
        inn_train = [_u[slice_inn] for _u in u_train]
        inn_val = [_u[slice_inn] for _u in u_val]
        data.update({'_slice_inn': slice_inn})

    elif hasattr(cfg, 'pressure_inlet_slice'):
        inn_loc = slice_from_tuple(cfg.pressure_inlet_slice)
        s_pressure = (np.s_[:],) + inn_loc + (np.s_[-1],)
        inn_train = [_u[s_pressure] for _u in u_train]
        inn_val = [_u[s_pressure] for _u in u_val]

    else:
        logger.critical('Pressure input is not defined in config. Please define inputs.')
        raise NotImplementedError

    data.update({
        'u_train': u_train,
        'u_val': u_val,
        'inn_train': inn_train,
        'inn_val': inn_val,
        'sets_index': sets_index,
    })


    return data, datainfo


def dataloader_2dkol(cfg:ConfigDict|None = None) -> tuple[dict,ClassDataMetadata]:
    '''Load 2D Kolmogorov flow generated with KolSol.'''
    if not cfg:
        cfg = FLAGS.cfg.data_config
    if cfg.remove_mean:
        warnings.warn('Method of removing mean from the Kolmogorov data has not been implemented. Ignoring remove_mean in configs.')
    
    data, datainfo = _load_kolsol(cfg,2)


    ngrid = data['u_train'].shape[datainfo.axx]
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


    ngrid = data['u_train'].shape[datainfo.axx]
    f = simulation.kolsol_forcing_term(cfg.forcing_frequency,ngrid,3)
    data.update({'forcing': f})
    
    return data, datainfo


def dataloader_3dkolsets(cfg:ConfigDict|None = None) -> tuple[dict, ClassDataMetadata]:
    if not cfg:
        cfg = FLAGS.cfg.data_config
    if cfg.remove_mean:
        warnings.warn('Method of removing mean from the Kolmogorov data has not been implemented. Ignoring remove_mean in configs.')
    
    data, datainfo = _load_kolsol(cfg, 3, multiplesets=True)

    ngrid = data['u_train'][0].shape[datainfo.axx]
    f = simulation.kolsol_forcing_term(cfg.forcing_frequency,ngrid,3)
    data.update({'forcing': f[:,:,:,:,0:1]})
    
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
    [u], _ = data_partition(
        x, 
        axis=0, 
        partition=[cfg.nsample],
        REMOVE_MEAN=cfg.remove_mean,
        randseed=randseed,
        shuffle=cfg.shuffle
    )

    ## get inputs
    inn_loc = slice_from_tuple(cfg.pressure_inlet_slice)
    s_pressure = (np.s_[:],) + inn_loc + (np.s_[-1],)
    logger.info(f'Taking input preessure measurememts at {s_pressure}.')
    inn = u[s_pressure].reshape((cfg.nsample,-1))
    
    ## Batching and take validation set 
    u = batching(cfg.batch_size, u)
    inn = batching(cfg.batch_size, inn)
    u_train, u_val = [], []
    inn_train, inn_val = [], []
    _idx = np.array(list(range(len(u))))
    _idx_val = _idx[list(cfg.val_batch_idx)]
    for i in _idx:
        if i in _idx_val:
            u_val.append(u[i])
            inn_val.append(u[i])
        else:
            u_train.append(u[i])
            inn_train.append(u[i])
    logger.debug(f"{np.sum([_u.shape[0] for _u in u_val])} snapshots are used for validation.")

    data.update({
        'u_train': u_train, # [t,x,y,4]
        'u_val': u_val, # [t,x,y,4]
        'inn_train': inn_train, 
        'inn_val': inn_val
    })

    logger.debug(f'Shape of the training set:{u_train[0].shape}, shape of the inputs:{inn_train[0].shape}')

    return data, datainfo




def batching(batch_size:int, data:Array, sets_index:list[int] = None) -> list[Array]:
    '''Split data into nb_batches number of batches along axis 0.'''
    if sets_index is not None:
        _index = np.cumsum(sets_index)
        _index = np.insert(_index,0,0)
        logger.debug(f'Dataset index {list(_index)}. The number of snapshots in total is {data.shape[0]}, {batch_size} snapshots per batch.')
        batched = []
        for i in range(1,len(_index)):
            n_equal_size = sets_index[i-1] // batch_size
            if n_equal_size == 0:
                logger.error(f'There are {sets_index[i-1]} snapshots in this dataset but the user asked for {batch_size} snapshots per batch.')
            if sets_index[i-1] % batch_size != 0:
                logger.warning('The batches do not have equal numbers of batches')
            n = [batch_size]*n_equal_size
            n = np.cumsum(n)[:-1]
            batched.extend(
                np.split(
                    data[_index[i-1]:_index[i],...],
                    n,
                    axis=0
                )
            )
        logger.debug(f'Batch sizes are {[d.shape[0] for d in batched]}')
        return batched
    else:        
        nb_batches = data.shape[0] // batch_size
        if data.shape[0] % nb_batches != 0:
            logger.warning('The batches do not have equal numbers of batches')
        return np.array_split(data,nb_batches,axis=0)


def _add_whitenoise(x:Array, randseed:int, cfg:ConfigDict, sets_index:list|None = None) -> tuple[Array,Array,Array]:
    logger.info('Adding white noise to data.')

    [u], _ = data_partition(
        x, 
        axis=0, 
        partition=[cfg.nsample],
        REMOVE_MEAN=cfg.remove_mean,
        randseed=randseed,
        shuffle=cfg.shuffle
    )

    ## add white noise
    rng = np.random.default_rng(randseed)
    std_data = np.std(u,axis=tuple(np.arange(u.ndim-1)),ddof=1)
    std_n = get_whitenoise_std(cfg.snr,std_data)
    noise = rng.normal([0]*len(std_n),std_n,size=u.shape)
    x = u + noise
    
    ## Batching and take validation set. These are clean data
    u_batched = batching(cfg.batch_size, u, sets_index)
    u_train_clean, u_val_clean = [], []
    _idx = np.array(list(range(len(u_batched))))
    _idx_val = _idx[list(cfg.val_batch_idx)]
    for i in _idx:
        if i in _idx_val:
            u_val_clean.append(u_batched[i])
        else:
            u_train_clean.append(u_batched[i])

    return x, u_train_clean, u_val_clean