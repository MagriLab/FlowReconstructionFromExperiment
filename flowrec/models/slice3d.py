import logging
logger = logging.getLogger(f'fr.{__name__}')
import jax
import jax.numpy as jnp
import haiku as hk
import warnings

from ._general import BaseModel
from .._typing import *
from ..training_and_states import params_merge
from ..training_and_states import params_split as params_split_general

from typing import Optional, Callable, Sequence, List, Tuple
from jax.tree_util import Partial



class Slice3D(hk.Module):
    '''2D network trained on 3D data.'''

    def __init__(
            self,
            pretrained_model: hk.Module, # model
            newvar_model:hk.Module,
            pretrained_config:dict, 
            newvar_config:dict,
            map_axis:Tuple[int,int], # map which input axis to which output axis 
            reduce_layers:Sequence = [10,], # int for linear
            pretrain_shape:Tuple = (64,64,3), # The shape of output from pretrained_model
            newvar_shape:Tuple = (64,64,1), # The shape of the new variables to add to the output from pretrained_model, both shapes should be 2D with number of variables as the last number
            activation:str|Callable[[jnp.ndarray],jnp.ndarray]=jax.nn.tanh,
            name:Optional[str] = 'slice3d',
            dropout_rate:Optional[float] = None,
    ):
        super().__init__(name=name)
        self.pretrain_mdl = pretrained_model(**pretrained_config)
        self.newvar_mdl = newvar_model(**newvar_config)
        if isinstance(activation,str):
            self.act = getattr(jax.nn,activation)
        else:
            self.act = activation
        self.map_axis = map_axis
        self.pretrain_shape = pretrain_shape
        self.newvar_shape = newvar_shape
        if len(newvar_shape) != len(pretrain_shape):
            raise ValueError('The pretrain and newvar shapes are not compatible') 
        self.dropout_rate = dropout_rate

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
                out1_reduce = self.dropout(out1_reduce, training, self.dropout_rate)
                out1_reduce = l(out1_reduce)
            out2 = self.merge_layer(x1)
            out2 = out2 + out1_reduce
            logger.debug(f'Input to the newvar model has shape {out2.shape}.')
            out2 = self.newvar_mdl(out2, training=training)
            logger.debug(f'Output of the newvar model has shape {out2.shape}.')
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
    Has methods:
        - init: same as haiku.Transformed.init.
        - apply: same as haiku.Transformed.apply.
        - predict: apply in prediction mode.
        - load_pretrained_weights: load the pre-trained model weights into the new model weights when the layers have the same name.
        - params_split: split the params into trainable and non-trainable
        - set_nontrainable: make the model remember the non-trainable weights
        - apply_trainable: apply the model using the trianable weights and the previously memorised non-trainable weights.

    To fine-tune this model starting from a pre-trained model:  
    init -> load_pretrained_weights -> params_split -> set_nontrainable -> apply_trainable
    ''' 

    def __init__(self, name='slice3d', **kwargs):
        super().__init__()
        if name is None:
            logger.eror('Model name cannot be none. Setting it to slice3d.')
            self.name = 'slice3d'
        else:
            self.name = name

        def forward(x,training=True):
            mdl = Slice3D(name=self.name,**kwargs)
            return mdl(x, training)

        self.mdl = hk.transform(forward)
        self._pretrain_mdl = kwargs['pretrained_model']
        self._pretrain_config = kwargs['pretrained_config']

        self._apply = jax.jit(self.mdl.apply,static_argnames=['training'])
        self._init = jax.jit(self.mdl.init)
        self._predict = jax.jit(jax.tree_util.Partial(self.mdl.apply,training=False))

    def init(self, rng, sample_input):
        '''Initialise params'''
        params = self._init(rng, sample_input)
        return params

    def apply(self, params:hk.Params, rng:jax.random.PRNGKey, *args, **kwargs):
        '''hk.Transformed.apply, training mode by default\n
        
        Arguments:\n
            params: hk.Params.\n
            rng: jax random number generator key.\n
            Also takes positional and keyword arguments for hk.Transformed.apply.
        '''
        return self._apply(params, rng, *args, **kwargs)
    
    def _check_layer_names(self, paramsnames, oldparamsnames):
        noprefix_match = any(name in paramsnames for name in oldparamsnames)
        prefix_match = any(f'{self.name}/~/{name}' in paramsnames for name in oldparamsnames)
        if (noprefix_match and prefix_match):
            raise ValueError('The Frozen layer names are not unique.')
        if noprefix_match:
            matching_param_names = oldparamsnames
        else:
            matching_param_names = [f'{self.name}/~/{name}' for name in oldparamsnames]

        return prefix_match, matching_param_names

    
    def load_old_weights(self, params: hk.Params, old_params: hk.Params) -> hk.Params:
        '''Loading the old model weights into the new weights for layers with the same name.
        ------------------------------
        For example, if the old model has a layer 'linear0', and the new Slice3D model has a layers 'slice3d/\~/linear0' and 'slice3d/\~/linear1'. `self.load_pretrained_weights(params, pretrained_params)` will load the pre-trained 'linear0' into 'slice3d/\~/linear0'.
        '''
        prefix_match, _ = self._check_layer_names(list(params), list(old_params))
        for k, layer in old_params.items():
            if prefix_match:
                k1 = f'{self.name}/~/' + k
            else:
                k1 = k
            if k1 in params.keys():
                params[k1].update(layer)
                logger.debug(f'Loading weights from {k1}')
        return params
    
    def freeze_layers(self, params:hk.Params, frozen_layer_names:List[str]) -> tuple[hk.Params, hk.Params]:
        '''Split params into non-trainable and trainable.'''
        _, layer_names = self._check_layer_names(list(params), frozen_layer_names)
        logger.debug(f'Freezeing layers {layer_names}')
        frozen_params, trainable_params = params_split_general(params, layer_names)
        return frozen_params, trainable_params

    def set_nontrainable(self, non_trainable_params:hk.Params):
        '''Make the model remeber the non_trainable_params'''
        if hasattr(self, 'params_merge'):
            warnings.warn('Overriding the old non_trainable_params.')
        self.params_merge = Partial(params_merge, non_trainable_params)
    
    def apply_trainable(self, trainable_params:hk.Params, rng:jax.random.PRNGKey, *args, **kwargs):
        '''Apply the model using the trainable_params.
        ------------------------------
        Must call `set_nontrainable(non_trainable_params)` first.
        '''
        params = self.params_merge(trainable_params)
        return self._apply(params, rng, *args, **kwargs)

    def predict(self,params:hk.Params,x,**kwargs):
        '''Same as apply, but Training flag is False and no randomness.'''
        return self._predict(params,None,x,**kwargs)
    
    @property
    def get_pretrain_model(self):
        def forward_pretrain(x,training=False):
            mdl_pretrain = self._pretrain_mdl(**self._pretrain_config)
            return mdl_pretrain(x, training)
        return hk.transform(forward_pretrain)
