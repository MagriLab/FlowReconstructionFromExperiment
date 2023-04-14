from flowrec._typing import *
from ml_collections.config_dict import ConfigDict

from flowrec.models import cnn, feedforward

import logging
logger = logging.getLogger(f'fr.{__name__}')


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

    def prep_data(data:dict) -> dict:
        # make data into suitable form
        # data.update({'u_train':new_u_train,'inn_train':new_inn_train})
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
