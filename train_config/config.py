from ml_collections import config_dict
from ml_collections.config_dict import placeholder
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

    cfg.data_config.data_dir = './local_data/re100/'
    cfg.data_config.shuffle = False
    cfg.data_config.randseed = placeholder(int)
    cfg.data_config.remove_mean = False
    cfg.data_config.normalise = True
    cfg.data_config.train_test_split = [600,100,100]
    cfg.data_config.re = 100.0
    cfg.data_config.dt = 0.125
    cfg.data_config.dx = 12/512
    cfg.data_config.dy = 4/128
    cfg.data_config.dz = placeholder(float)
    cfg.data_config.snr = placeholder(float)


    ## Model
    cfg.model_config = config_dict.ConfigDict()
    
    cfg.model_config.dropout_rate = 0.0
    

    ## Training
    cfg.train_config = config_dict.ConfigDict()
    
    cfg.train_config.nb_batches = 10
    cfg.train_config.learning_rate = 3e-4
    cfg.train_config.regularisation_strength = 0.0
    cfg.train_config.epochs = 20000

    cfg.train_config.randseed = placeholder(int)



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
    _observe = 'grid'
    _select_model = 'ffcnn'
    _loss_fn = 'physicswithdata'


    cfg = get_basic_config()


    ## get functions from options
    if 'dataloader' in user:
        _dataloader = user['dataloader']
    if 'observe' in user:
        _observe = user['observe']
    if 'select_model' in user:
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
    if _dataloader == '2dtriangle':
        cfg.data_config.update({
            'slice_to_keep': ((None,), (None,), (None,250,None), (None,)),
        })
    else:
        raise ValueError('Invalid dataloader option.')


    if _observe == 'grid':
        cfg.data_config.update({
            'slice_grid_sensors': ((None,None,15), (None,None,5))
        }) # spatial slicing, the default is equivalent to np.s_[::15,::5] in x and y direction
    elif _observe == 'grid_pin':
        cfg.data_config.update({
            'slice_grid_sensors': ((None,None,15), (None,None,5)),
        })

        if _dataloader == '2dtriangle':
            cfg.data_config.update({
                'pressure_inlet_slice': ((0,1,None),(49,80,None))
            })
        else:
            cfg.data_config.update({
                'pressure_inlet_slice': placeholder(tuple)
            })
    elif _observe == 'sparse':
        cfg.data_config.update({
            'sensor_index': placeholder(tuple)
        })
    elif _observe == 'sparse_pin':
        cfg.data_config.update({
            'sensor_index': placeholder(tuple)
        })
        if _dataloader == '2dtriangle':
            cfg.data_config.update({
                'pressure_inlet_slice': ((0,1,None),(49,80,None))
            })
        else:
            cfg.data_config.update({
                'pressure_inlet_slice': placeholder(tuple)
            })
    else:
        raise ValueError('Invalid observe option.')
    

    if _select_model == 'ffcnn':
        cfg.model_config.update({
            'mlp_layers': (96750,),
            'output_shape': (250,129,3),
            'cnn_channels': (32,16,3),
            'cnn_filters': ((3,3),),
        })
    else:
        raise ValueError('Invalid select_model option.')


    if _loss_fn == 'physicswithdata' or 'physicsreplacemean':
        cfg.train_config.update({
            'weight_momentum': 1.0,
            'weight_continuity': 1.0,
            'weight_sensors': 0.0
        })
    elif _loss_fn == 'physicsnoreplace' or 'physicsandmean':
        cfg.train_config.update({
            'weight_momentum': 1.0,
            'weight_continuity': 1.0,
            'weight_sensors': 10.0
        })
    else:
        raise ValueError('Invalid loss_fn option.')



    return cfg
