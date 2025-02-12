import logging
logger = logging.getLogger(f'fr.{__name__}')
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from ._general import BaseModel
from .._typing import *

from typing import Optional, Callable, Sequence, List, Tuple
from jax.tree_util import Partial



class Slice3D(hk.Module):
    '''2D network trained on 3D data.'''

    def __init__(
            self,
            pretrained_model: hk.Module, # model
            newvar_model:hk.Module,
            pretrained_config:dict = {}, 
            newvar_config:dict = {},
            reduce_layers:Sequence = [], #int for linear
            map_axis:Tuple[int,int] = (2,3), # map which input axis to which output axis  
            pretrain_shape:Tuple = (64,64,3), # The shape of output from pretrained_model
            newvar_shape:Tuple = (64,64,1), # The shape of the new variables to add to the output from pretrained_model, both shapes should be 2D with number of variables as the last number
            activation=jax.nn.tanh,
            name:Optional[str] = 'slice3d',
    ):
        super().__init__(name=name)
        self.pretrain_mdl = pretrained_model(**pretrained_config)
        self.newvar_mdl = newvar_model(**newvar_config)
        self.act = activation
        self.map_axis = map_axis
        self.pretrain_shape = pretrain_shape
        self.newvar_shape = newvar_shape
        if len(newvar_shape) != len(pretrain_shape):
            raise ValueError('The pretrain and newvar shapes are not compatible') 

        # Define the layers used to reduce the pre-trained output to merge into the new branches
        self.reduce_layers = [hk.Linear(n, name=f'reduce{i}') for i,n in enumerate(reduce_layers)]
        self.merge_layer = hk.Linear(reduce_layers[-1], name=f'merge')


    def __call__(self, x, training):
        # First define the network for a slice, then vmap over the direction defined in self.map_axis
        def over_axis(x1,training):
            out1 = self.pretrain_mdl(x1, training=training)
            # the reduce layer 
            out1_reduce = jnp.copy(out1)
            for l in self.reduce_layers:
                out1_reduce = self.act(out1_reduce)
                out1_reduce = l(out1_reduce)
            out2 = self.merge_layer(x1)
            out2 = out2 + out1_reduce
            print(out2.shape)
            out2 = self.newvar_mdl(out2, training=training)
            print(out2.shape)
            out1 = out1.reshape((-1,) + self.pretrain_shape)
            out2 = out2.reshape((-1,) + self.newvar_shape)
            return jnp.concatenate((out1,out2), axis=-1)
        
        out = jax.vmap(over_axis, (self.map_axis[0],None), self.map_axis[1])(x, training)
        return out

    @staticmethod
    def dropout(x, training:bool, dropout_rate:Optional[float]):
        """Apply dropout with if training is True"""
        if training and (dropout_rate is not None):
            logger.debug('Doing dropout')
            return hk.dropout(hk.next_rng_key(), dropout_rate, x)
        else:
            return x






class Model(BaseModel):
    '''
    Has methods:\n
        init: same as haiku.Transformed.init.\n
        apply: same as haiku.Transformed.apply.\n
        predict: apply in prediction mode.
    ''' 
