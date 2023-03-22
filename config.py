from ml_collections import config_dict
from ml_collections.config_dict import placeholder
from typing import Callable
from absl import logging
import train_options



def get_basic_config() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()


    ## Case
    cfg.case = config_dict.ConfigDict()

    cfg.case.dataloader = getattr(train_options,'dataloader_2dtriangle')
    cfg.case.observe = getattr(train_options, 'observe_grid')
    cfg.case.select_model = getattr(train_options, 'select_model_ffcnn')
    cfg.case.loss_fn = getattr(train_options, 'loss_fn_physicswithdata')


    ## Data
    cfg.data_config = config_dict.ConfigDict()

    cfg.data_config.data_dir = './local_data/re100/'
    cfg.data_config.shuffle = False
    cfg.data_config.randseed = placeholder(int)
    cfg.data_config.remove_mean = False
    cfg.data_config.re = 100
    cfg.data_config.dt = 0.125
    cfg.data_config.dx = 12/512
    cfg.data_config.dy = 4/128
    cfg.data_config.dz = placeholder(float)


    ## Model
    cfg.model_config = config_dict.ConfigDict()

    cfg.model_config.dropout_rate = 0.0
    

    ## Training
    cfg.train_config = config_dict.ConfigDict()
    
    cfg.train_config.nb_batches = placeholder(int)
    cfg.train_config.weight_physics = 0.9
    cfg.train_config.weight_sensors = 0.1
    cfg.train_config.learning_rate = 3e-4
    cfg.train_config.epochs = 20000



    return cfg









def get_config(cfgstr:str = None):
    # example cfgstr = 'dataloader@2dtriangle,model@ffcnn'
    
    if cfgstr:
        user = dict([x.split('@') for x in cfgstr.split(',')])
    else:
        user = None
        logging.warning('No training case is selected, proceeds with the basic configuration.\n Are you sure this is not a mistake?')

    # Set up default options
    _dataloader = '2dtriangle'
    _observe = 'grid'
    _select_model = 'ffcnn'
    _loss_fn = 'physicswithdata'


    cfg = get_basic_config()


    ## get functions from options
    if 'dataloader' in user:
        _dataloader = user['dataloader']
        cfg.case.dataloader = getattr(train_options, f'dataloader_{_dataloader}')
    if 'observe' in user:
        _observe = user['observe']
        cfg.case.observe = getattr(train_options, f'observe_{_observe}')
    if 'select_model' in user:
        _select_model = user['select_model']
        cfg.case.select_model = getattr(train_options, f'select_model_{_select_model}')
    if 'loss_fn' in user:
        _loss_fn = user['loss_fn']
        cfg.case.loss_fn = getattr(train_options, f'loss_fn_{_loss_fn}')



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
    else:
        raise ValueError('Invalid observe option.')
    

    if _select_model == 'ffcnn':
        cfg.model_config.update({
            'mlp_layers': ()
        })
    else:
        raise ValueError('Invalid select_model option.')


    if _loss_fn == 'physicswithdata':
        pass
    else:
        raise ValueError('Invalid loss_fn option.')



    return cfg
