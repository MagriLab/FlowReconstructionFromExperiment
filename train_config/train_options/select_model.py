from flowrec._typing import *
from ml_collections.config_dict import ConfigDict

from flowrec.models import cnn, feedforward
from flowrec.data import normalise

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

    def prep_data(data:dict) -> dict:
        # make data into suitable form
        return data
    
    def make_model(model_config:ConfigDict) -> BaseModel:
        # mdl = 'a BaseModel createed with parameters in model_config'
        # return mdl
        pass

    return prep_data, make_model
        




def select_model_ffcnn(**kwargs):

    def prep_data(data:dict, **kwargs) -> dict:
        '''make data into suitable form
        data.update({'u_train':new_u_train,'inn_train':new_inn_train})'''

        if ('normalise' in kwargs) and kwargs['normalise'] is True:

            r_train = data['train_minmax']
            r_val = data['val_minmax']

            [new_train_inn, new_val_inn], _ = normalise(data['inn_train'], data['inn_val'], range=[r_train[-1],r_val[-1]])
            data.update({
                'inn_train': new_train_inn,
                'inn_val': new_val_inn
            })
            logger.debug('Update inputs to normalised inputs.')

            if data['u_train_clean'] is not None:
                u_train = data['u_train_clean']
                u_val = data['u_val_clean']
                num_components = u_train.shape[-1]

                x_train_components = np.squeeze(np.split(u_train, num_components, axis=-1))
                x_val_components = np.squeeze(np.split(u_val, num_components, axis=-1))
                x_train_normalised, _ = normalise(*x_train_components, range=r_train)
                x_val_normalised, _ = normalise(*x_val_components, range=r_val)
                u_train = np.stack(x_train_normalised,axis=-1)
                u_val = np.stack(x_val_normalised,axis=-1)

                data.update({
                    'u_train_clean': u_train,
                    'u_val_clean': u_val
                })
                logger.debug('Update clean data to normalised clean data.')


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
