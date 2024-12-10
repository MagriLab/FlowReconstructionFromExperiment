from ml_collections import config_dict
from ml_collections.config_dict import placeholder, required_placeholder
import numpy as np
import warnings
import train_config.train_options.dataloader as dataloaderoptions
import train_config.train_options.observe as observeoptions
import train_config.train_options.select_model as modeloptions
import train_config.train_options.loss_fn as lossfnoptions

def _undefined_function():
    pass


def get_basic_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()


    ## Case
    cfg.case = config_dict.ConfigDict()

    cfg.case.dataloader = _undefined_function
    cfg.case.observe = _undefined_function
    cfg.case.select_model = _undefined_function
    cfg.case.loss_fn = _undefined_function
    cfg.case._case_dataloader = placeholder(str)
    cfg.case._case_observe = placeholder(str)
    cfg.case._case_select_model = placeholder(str)
    cfg.case._case_loss_fn = placeholder(str)


    ## Data
    cfg.data_config = config_dict.ConfigDict()

    cfg.data_config.data_dir = placeholder(str)
    cfg.data_config.shuffle = False
    cfg.data_config.randseed = placeholder(int)
    cfg.data_config.remove_mean = False # Do NOT remove mean
    cfg.data_config.normalise = False
    cfg.data_config.train_test_split = placeholder(tuple)
    cfg.data_config.re = placeholder(float)
    cfg.data_config.dt = placeholder(float)
    cfg.data_config.dx = placeholder(float)
    cfg.data_config.dy = placeholder(float)
    cfg.data_config.dz = placeholder(float)
    cfg.data_config.snr = placeholder(float)
    cfg.data_config.filter = placeholder(str)


    ## Model
    cfg.model_config = config_dict.ConfigDict()
    
    cfg.model_config.dropout_rate = 0.0
    

    ## Training
    cfg.train_config = config_dict.ConfigDict()
    
    cfg.train_config.nb_batches = 20
    cfg.train_config.learning_rate = 3e-4
    cfg.train_config.regularisation_strength = 0.0
    cfg.train_config.epochs = 20000

    cfg.train_config.randseed = placeholder(int)
    
    cfg.train_config.gradient_clip = placeholder(float)
    cfg.train_config.lr_scheduler = 'constant'



    return cfg









def get_config(cfgstr:str = None):
    # example cfgstr = 'dataloader@2dtriangle,model@ffcnn'
    
    if cfgstr:
        user = dict([x.split('@') for x in cfgstr.split(',')])
    else:
        user = {}
        warnings.warn('No training case is selected, proceeds with the basic configuration.\n Are you sure this is not a mistake?')

    # Set up default options
    _dataloader = '2dtriangle'
    _observe = 'grid_pin'
    _select_model = 'fc2branch'
    _loss_fn = 'physicswithdata'


    cfg = get_basic_config()


    ## get functions from options
    if 'dataloader' in user:
        _dataloader = user['dataloader']
    if 'observe' in user:
        _observe = user['observe']
    if 'model' in user:
        _select_model = user['model']
    if 'loss_fn' in user:
        _loss_fn = user['loss_fn']

    cfg.case.update({
        'dataloader': getattr(dataloaderoptions, f'dataloader_{_dataloader}'),
        'observe': getattr(observeoptions, f'observe_{_observe}'),
        'select_model': getattr(modeloptions, f'select_model_{_select_model}'),
        'loss_fn': getattr(lossfnoptions, f'loss_fn_{_loss_fn}'),
        '_case_dataloader': _dataloader,
        '_case_observe': _observe,
        '_case_select_model': _select_model,
        '_case_loss_fn': _loss_fn
    })


    ## update config dictionary
    if _dataloader in _default_datacfg:
        cfg.data_config.update(_default_datacfg[_dataloader])
    else:
        raise ValueError('Invalid dataloader option.')


    if _observe == 'grid' or _observe == 'grid_pin':
        cfg.data_config.update({
            'slice_grid_sensors': ((None,None,15), (None,None,5))
        }) # spatial slicing, the default is equivalent to np.s_[::15,::5] in x and y direction
        if _dataloader == '3dvolvo':
            cfg.data_config.update({
                'slice_grid_sensors': ((None,None,None), (None,None,None),(10,11,None))
            }) # spatial slicing, the default is equivalent to np.s_[:,:,
    elif _observe == 'sparse' or _observe == 'sparse_pin':
        cfg.data_config.update({
            'sensor_index': placeholder(tuple)
        })
    elif _observe == 'random_pin':
        cfg.data_config.update({
            'random_sensors': placeholder(tuple), # (random seed, number of sensors)
            'sensor_index': placeholder(tuple),
        })
    else:
        raise ValueError('Invalid observe option.')
    

    if _select_model == 'ffcnn':
        cfg.model_config.update(_default_mdlcfg_ffcnn[_dataloader])
    elif _select_model == 'fc2branch':
        cfg.model_config.update(_default_mdlcfg_fc2branch[_dataloader])
    else:
        raise ValueError('Invalid select_model option.')

    _condition1 = ['physicswithdata', 'physicsreplacemean', 'physicswithdata_mae', 'physicsreplacemean_mae']
    _condition2 = ['physicsnoreplace', 'physicsandmean', 'physicsnoreplace_mae', 'physicsandmean_mae']
    if _loss_fn in _condition1:
        cfg.train_config.update({
            'weight_momentum': 1.0,
            'weight_continuity': 1.0,
            'weight_sensors': 0.0
        })
    elif _loss_fn in _condition2:
        cfg.train_config.update({
            'weight_momentum': 1.0,
            'weight_continuity': 1.0,
            'weight_sensors': 10.0
        })
    else:
        raise ValueError('Invalid loss_fn option.')



    return cfg


# =========================== Defaults =========================

_default_datacfg = {
    '2dtriangle': {
        'slice_to_keep': ((None,), (None,), (None,250,None), (None,)),
        'data_dir': './local_data/re100/',
        're': 100.0,
        'dt': 0.125,
        'dx': 12/512,
        'dy': 4/128,
        'pressure_inlet_slice': ((0,1,None),(49,80,None)),
        'train_test_split': (600,100,100)
    },
    '2dkol': {
        'data_dir': './local_data/kolmogorov/dim2_re34_k32_f4_dt1_grid128_25619.h5',
        're': 34.0,
        'dt': 0.1,
        'dx': 2*np.pi/128,
        'dy': 2*np.pi/128,
        'pressure_inlet_slice': placeholder(tuple),
        'random_input': placeholder(tuple), # (randseed, number of pressure sensors)
        'forcing_frequency': 4,
        'train_test_split': (6000,500,500)
    },
    '3dvolvo': {
        'data_dir': './local_data/volvorig/u166/',
        'pressure_inlet_slice': ((0,1,None),(None,None,2),(None,None,None)), # sensors at x=0, a slice at each z, sensor at every other y
        'train_test_split': (450, 40, 3)
    },
}


_default_mdlcfg_ffcnn = {
    '2dtriangle': {
        'mlp_layers': (96750,),
        'output_shape': (250,129,3),
        'cnn_channels': (32,16,3),
        'cnn_filters': ((3,3),),
    },
    '2dkol': {
        'mlp_layers': (49152,),
        'output_shape': (128,128,3),
        'cnn_channels': (3,3),
        'cnn_filters': ((3,3),),
    }
}
_default_mdlcfg_fc2branch = {
    '2dtriangle': {
        'img_shapes': ((128,64),(128,32),(64,16),(128,64),(250,129)),
        'b1_channels': (1,),
        'b2_channels': (8,16,8),
        'b3_channels': (4,3),
        'b1_filters': ((3,3),),
        'b2_filters': ((5,5),),
        'b3_filters': ((3,3),),
        'resize_method': 'linear',
        'fft_branch': False
    },
    '2dkol': {
        'img_shapes': ((64,64),(64,64),(32,32),(16,16),(32,32),(64,64),(128,128)),
        'b1_channels': (1,),
        'b2_channels': (4,8,16,8,4),
        'b3_channels': (4,3),
        'b1_filters': ((3,3),),
        'b2_filters': ((3,3),),
        'b3_filters': ((3,3),),
        'resize_method': 'linear',
        'fft_branch': True,
    },
    '3dvolvo': {
        'img_shapes': ((20,10,10),(10,10,10),(5,5,5),(10,10,10),(40,20,20)),
        'b1_channels': (1,),
        'b2_channels': (8,16,8),
        'b3_channels': (4,),
        'resize_method': 'linear',
        'fft_branch': False,
        'small_mlp': False,
    }
}
