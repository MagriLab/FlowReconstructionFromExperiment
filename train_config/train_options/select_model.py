import yaml
from flowrec._typing import *
from pathlib import Path
from ml_collections.config_dict import ConfigDict
from ast import literal_eval

import flowrec.signal as flowsignal
from flowrec.models import cnn, fourier2branch, feedforward, slice3d
from flowrec.data import normalise
from flowrec.training_and_states import restore_trainingstate


import logging
logger = logging.getLogger(f'fr.{__name__}')

from absl import flags
FLAGS = flags.FLAGS


def select_model_example(**kwargs):
    '''# Example model selection function.\n
    Pass select_model@example to use this model.\n

    Returns two functions -- prep_data and make_model.\n
    1. prep_data: (data:dict -> data_new:dict).
        Takes the data dictionary and performs data.update({'u_train':new_u_train,'inn_train':new_inn_train, 'u_val':new_u_val, 'inn_val':new_inn_val})
    
    2. make_model: (model_config -> BaseModel)
        Make a model with parameters in model_config and returns that model.

    '''

    def prep_data(data:dict, datainfo:DataMetadata, **kwargs) -> dict:
        # make data into suitable form
        return data
    
    def make_model(model_config:ConfigDict) -> BaseModel:
        # mdl = 'a BaseModel createed with parameters in model_config'
        # return mdl
        pass

    return prep_data, make_model
        




def select_model_ffcnn(**kwargs):
    if 'datacfg' in kwargs:
        flag_norm = kwargs['datacfg'].normalise
        filter_type = kwargs['datacfg'].filter
        

    def prep_data(data:dict, datainfo:DataMetadata, **kwargs) -> dict:
        '''make data into suitable form
        data.update({'u_train':new_u_train,'inn_train':new_inn_train})'''

        data = _flatten(data)

        if filter_type == 'lowpass':
            logger.critical('The losspass filter in dataloaders need to be updated to deal with batched input.')
            raise NotImplementedError
            logger.info('Using the lowpass filter to de-noise.')
            (nt_train, nin) = data['inn_train'].shape # always has shape [t,j]
            (nt_val, _) = data['inn_val'].shape
            fs = 1/datainfo.dt

            old_inn_train = data['inn_train']
            new_inn_train = np.zeros_like(old_inn_train)
            old_inn_val = data['inn_val']
            new_inn_val = np.zeros_like(old_inn_val)
            for n in range(nin):
                nest_t, pobserved_t = flowsignal.estimate_noise_floor(
                    old_inn_train[:,n],
                    fs,
                    window_size=int(0.005*nt_train),
                    start_idx=0
                )
                cutoff_t = flowsignal.estimate_noise_cutoff_frequency(
                    pobserved_t,
                    nest_t,
                    fs/nt_train,
                    0.1*fs/2
                )
                new_inn_train[:,n] = flowsignal.butter_lowpass_filter(
                    old_inn_train[:,n],
                    cutoff_t,
                    fs
                )

                nest_v, pobserved_v = flowsignal.estimate_noise_floor(
                    old_inn_val[:,n],
                    fs,
                    window_size=int(0.005*nt_val),
                    start_idx=0
                )
                cutoff_v = flowsignal.estimate_noise_cutoff_frequency(
                    pobserved_v,
                    nest_v,
                    fs/nt_val,
                    0.1*fs/2
                )
                new_inn_val[:,n] = flowsignal.butter_lowpass_filter(
                    old_inn_val[:,n],
                    cutoff_v,
                    fs
                ) 
            data.update({
                'inn_train': new_inn_train,
                'inn_val': new_inn_val
            })

            # has shape [t,...], ... changes when using different observation function
            shape_y_train = data['y_train'].shape 
            shape_y_val = data['y_val'].shape
            old_y_train = data['y_train'].reshape((nt_train,-1))
            new_y_train = np.zeros_like(old_y_train)
            old_y_val = data['y_val'].reshape((nt_val,-1))
            new_y_val = np.zeros_like(old_y_val)
            for n in range(old_y_train.shape[1]): # interate of all observations
                nest_t, pobserved_t = flowsignal.estimate_noise_floor(
                    old_y_train[:,n],
                    fs,
                    window_size=int(0.005*nt_train),
                    start_idx=0
                )
                cutoff_t = flowsignal.estimate_noise_cutoff_frequency(
                    pobserved_t,
                    nest_t,
                    fs/nt_train,
                    0.1*fs/2
                )
                new_y_train[:,n] = flowsignal.butter_lowpass_filter(
                    old_y_train[:,n],
                    cutoff_t,
                    fs
                )

                nest_v, pobserved_v = flowsignal.estimate_noise_floor(
                    old_y_val[:,n],
                    fs,
                    window_size=int(0.005*nt_val),
                    start_idx=0
                )
                cutoff_v = flowsignal.estimate_noise_cutoff_frequency(
                    pobserved_v,
                    nest_v,
                    fs/nt_val,
                    0.1*fs/2
                )
                new_y_val[:,n] = flowsignal.butter_lowpass_filter(
                    old_y_val[:,n],
                    cutoff_v,
                    fs
                )
            data.update({
                'y_train': new_y_train.reshape(shape_y_train),
                'y_val': new_y_val.reshape(shape_y_val),
            })
        elif filter_type is not None:
            logger.error('The requested filtering method is not implemented.')
            raise NotImplementedError
            

#         if ('normalise' in kwargs) and kwargs['normalise'] is True:
        data = _norm_inputs(flag_norm, data)

        return data


    def make_model(cfg:ConfigDict) -> BaseModel:
        mdl = cnn.Model(
            mlp_layers = cfg.mlp_layers,
            output_shape = cfg.output_shape,
            cnn_channels = cfg.cnn_channels,
            cnn_filters = cfg.cnn_filters,
            dropout_rate = cfg.dropout_rate
        )
        return mdl

    return prep_data, make_model



def select_model_fc2branch(**kwargs):
    '''# Example model selection function.\n
    Pass select_model@example to use this model.\n

    Returns two functions -- prep_data and make_model.\n
    1. prep_data: (data:dict -> data_new:dict).
        Takes the data dictionary and performs data.update({'u_train':new_u_train,'inn_train':new_inn_train, 'u_val':new_u_val, 'inn_val':new_inn_val})
    
    2. make_model: (model_config -> BaseModel)
        Make a model with parameters in model_config and returns that model.

    '''
    
    if 'datacfg' in kwargs:
        flag_norm = kwargs['datacfg'].normalise
        filter_type = kwargs['datacfg'].filter

    def prep_data(data:dict, datainfo:DataMetadata, **kwargs) -> dict:
        # make data into suitable form
        data = _flatten(data)        
        if filter_type == 'lowpass':
            logger.critical('The losspass filter in dataloaders need to be updated to deal with batched input.')
            raise NotImplementedError
            logger.info('Using the lowpass filter to de-noise.')
            (nt_train, nin) = data['inn_train'].shape # always has shape [t,j]
            (nt_val, _) = data['inn_val'].shape
            fs = 1/datainfo.dt

            old_inn_train = data['inn_train']
            new_inn_train = np.zeros_like(old_inn_train)
            old_inn_val = data['inn_val']
            new_inn_val = np.zeros_like(old_inn_val)
            for n in range(nin):
                nest_t, pobserved_t = flowsignal.estimate_noise_floor(
                    old_inn_train[:,n],
                    fs,
                    window_size=int(0.005*nt_train),
                    start_idx=0
                )
                cutoff_t = flowsignal.estimate_noise_cutoff_frequency(
                    pobserved_t,
                    nest_t,
                    fs/nt_train,
                    0.1*fs/2
                )
                new_inn_train[:,n] = flowsignal.butter_lowpass_filter(
                    old_inn_train[:,n],
                    cutoff_t,
                    fs
                )

                nest_v, pobserved_v = flowsignal.estimate_noise_floor(
                    old_inn_val[:,n],
                    fs,
                    window_size=int(0.005*nt_val),
                    start_idx=0
                )
                cutoff_v = flowsignal.estimate_noise_cutoff_frequency(
                    pobserved_v,
                    nest_v,
                    fs/nt_val,
                    0.1*fs/2
                )
                new_inn_val[:,n] = flowsignal.butter_lowpass_filter(
                    old_inn_val[:,n],
                    cutoff_v,
                    fs
                ) 
            data.update({
                'inn_train': new_inn_train,
                'inn_val': new_inn_val
            })

            # has shape [t,...], ... changes when using different observation function
            shape_y_train = data['y_train'].shape 
            shape_y_val = data['y_val'].shape
            old_y_train = data['y_train'].reshape((nt_train,-1))
            new_y_train = np.zeros_like(old_y_train)
            old_y_val = data['y_val'].reshape((nt_val,-1))
            new_y_val = np.zeros_like(old_y_val)
            for n in range(old_y_train.shape[1]): # interate of all observations
                nest_t, pobserved_t = flowsignal.estimate_noise_floor(
                    old_y_train[:,n],
                    fs,
                    window_size=int(0.005*nt_train),
                    start_idx=0
                )
                cutoff_t = flowsignal.estimate_noise_cutoff_frequency(
                    pobserved_t,
                    nest_t,
                    fs/nt_train,
                    0.1*fs/2
                )
                new_y_train[:,n] = flowsignal.butter_lowpass_filter(
                    old_y_train[:,n],
                    cutoff_t,
                    fs
                )

                nest_v, pobserved_v = flowsignal.estimate_noise_floor(
                    old_y_val[:,n],
                    fs,
                    window_size=int(0.005*nt_val),
                    start_idx=0
                )
                cutoff_v = flowsignal.estimate_noise_cutoff_frequency(
                    pobserved_v,
                    nest_v,
                    fs/nt_val,
                    0.1*fs/2
                )
                new_y_val[:,n] = flowsignal.butter_lowpass_filter(
                    old_y_val[:,n],
                    cutoff_v,
                    fs
                )
            data.update({
                'y_train': new_y_train.reshape(shape_y_train),
                'y_val': new_y_val.reshape(shape_y_val),
            })
        elif filter_type is not None:
            logger.error('The requested filtering method is not implemented.')
            raise NotImplementedError

#       if ('normalise' in kwargs) and kwargs['normalise'] is True:
        data = _norm_inputs(flag_norm, data)

        return data
    
    def make_model(cfg:ConfigDict) -> BaseModel:
        # mdl = 'a BaseModel createed with parameters in model_config'
        # return mdl
        mdl_config_dict = cfg.to_dict()
        mdl = fourier2branch.Model(**mdl_config_dict)
        return mdl

    return prep_data, make_model
    

def select_model_ff(**kwargs):
    '''Simple feedforward model.\n
    Pass select_model@ff to use this model.\n

    Returns two functions -- prep_data and make_model.\n
    1. prep_data: (data:dict -> data_new:dict).
        Takes the data dictionary and performs data.update({'u_train':new_u_train,'inn_train':new_inn_train, 'u_val':new_u_val, 'inn_val':new_inn_val})
    
    2. make_model: (model_config -> BaseModel)
        Make a model with parameters in model_config and returns that model.

    '''

    if 'datacfg' in kwargs:
        flag_norm = kwargs['datacfg'].normalise
        filter_type = kwargs['datacfg'].filter

    def prep_data(data:dict, datainfo:DataMetadata, **kwargs) -> dict:
        # make data into suitable form
        if filter_type is not None:
            logger.critical('The requested filtering method is not implemented for this model.')
            raise NotImplementedError
        data = _norm_inputs(flag_norm, data)
        ## Flatten data for FF model
        keys_to_flatten = ['y_train', 'y_val', 'u_train_clean', 'u_val_clean', 'u_train', 'u_val', 'inn_train', 'inn_val']
        data = _flatten(data, keys_to_flatten)
        return data
    
    def make_model(model_config:ConfigDict) -> BaseModel:
        # mdl = 'a BaseModel createed with parameters in model_config'
        # return mdl
        mdl = feedforward.Model(**model_config)
        return mdl

    return prep_data, make_model
        

def select_model_slice3d(**kwargs):
    '''3D volume from 2D slice model.\n
    Pass select_model@slice3d to use this model.\n

    Returns two functions -- prep_data and make_model.\n
    1. prep_data: (data:dict -> data_new:dict).
        Takes the data dictionary and performs data.update({'u_train':new_u_train,'inn_train':new_inn_train, 'u_val':new_u_val, 'inn_val':new_inn_val})
    
    2. make_model: (model_config -> BaseModel)
        Make a model with parameters in model_config and returns that model.

    '''
    if 'datacfg' in kwargs:
        flag_norm = kwargs['datacfg'].normalise
        filter_type = kwargs['datacfg'].filter
        if filter_type is not None:
            logger.error('The requested filtering method is not implemented.')
            raise NotImplementedError
    map_axis = kwargs['mdlcfg'].map_axis

    def prep_data(data:dict, datainfo:DataMetadata, **kwargs) -> dict:
        # make data into suitable form
        new_inn = [np.squeeze(inn) for inn in data['inn_train']]
        if map_axis[0] >= new_inn[0].ndim:
            raise ValueError(f'Cannot map input axis {map_axis[0]} to output axis {map_axis[1]} because the input only have {new_inn[0].ndim} dimensions.')
        new_inn_val = [np.squeeze(inn) for inn in data['inn_val']]
        data.update({
            'inn_train': new_inn,
            'inn_val': new_inn_val,
        })
        return data
    
    def make_model(model_config:ConfigDict) -> BaseModel:
        
        mdlcfg_dict = model_config.to_dict()
        
        model_options = {
            'ffcnn': cnn.MLPWithCNN,
            'ff': feedforward.MLP,
            'fc2branch': fourier2branch.Fourier2Branch,
        }
        if mdlcfg_dict['pretrain_config'] is not None: # Prioritize user inputs
            pretrained_config = literal_eval(mdlcfg_dict['pretrain_config'])
            pretrained_model = model_options[mdlcfg_dict['pretrain_model']]
        elif model_config.load_pretrain_config is not None:
            load_state_from = Path(model_config.load_pretrain_config)
            logger.info(f"Loading pretrained model config from {Path(load_state_from, 'config.yml')}")
            with open(Path(load_state_from, 'config.yml'), 'r') as f:
                old_config = yaml.load(f, Loader=yaml.UnsafeLoader)
            pretrained_config = old_config.model_config.to_dict()
            pretrained_model = model_options[old_config.case._case_select_model]
        else:
            raise ValueError('Cannot find pretrain config and pretrain model.')
        logger.debug(f'Pretrained model: \n        model: {old_config.case._case_select_model}\n        config: {pretrained_config}')
        newvar_model = model_options[mdlcfg_dict['newvar_model']]
        newvar_config = literal_eval(mdlcfg_dict['newvar_config'])
        logger.debug(f'Newvar model config: {newvar_config}')

        for k in ['pretrain_config', 'pretrain_model', 'newvar_model', 'newvar_config', 'load_pretrain_config']:
            del mdlcfg_dict[k]

        mdl = slice3d.Model(
            pretrained_model = pretrained_model,
            newvar_model = newvar_model,
            pretrained_config = pretrained_config,
            newvar_config = newvar_config,
            **mdlcfg_dict
        )
        return mdl

    return prep_data, make_model
        


# =========== helper function ===================
def _norm_inputs(flag_norm:bool, data:dict):
    if flag_norm:

        r_train = data['train_minmax']
        r_val = data['val_minmax']
        
        logger.info('Normalising inputs to the network.')
        
        new_train_inn, _ = normalise(*data['inn_train'], range=[r_train[-1]])
        new_val_inn, _ = normalise(*data['inn_val'], range=[r_val[-1]])

        logger.debug('Update inputs to normalised inputs.')

        data.update({
            'inn_train': new_train_inn,
            'inn_val': new_val_inn,
        })
    return data

def _flatten(data, input_keys = ['inn_train', 'inn_val']):
    logger.debug('Flattening data')
    for k in input_keys:
        modified = data[k]
        modified = [a.reshape((a.shape[0],-1)) for a in modified]
        data.update({
            k: modified
        })
    return data