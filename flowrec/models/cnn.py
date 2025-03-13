import logging
logger = logging.getLogger(f'fr.{__name__}')
import warnings

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from .feedforward import MLP
from ._general import BaseModel

from typing import Optional, Callable, Sequence, List


class MLPWithCNN(hk.Module):
    ''' MLP feeding into a CNN network. 
    The CNN network does not change the size of the image.\n
    '''

    def __init__(self, 
                mlp_layers:Sequence[int],
                output_shape:Sequence[int],
                cnn_channels:Sequence[int],
                cnn_filters:Sequence[tuple],
                activation:Callable[[jnp.ndarray],jnp.ndarray] = jax.nn.tanh,
                w_init:Optional[hk.initializers.Initializer]=hk.initializers.VarianceScaling(1.0,"fan_avg","uniform"),
                dropout_rate:Optional[float] = None,
                mlp_kwargs:dict = {},
                conv_kwargs:dict = {},
                name: Optional[str] = 'ffcnn'):
        '''Initialise with:\n
            mlp_layers: list of output sizes of all layers in the MLP.\n
            output_shape: list of dimensions of the target output.\n
            cnn_channels: list of number of output channels, each corresponds to one convolution layer. \n
            cnn_filters: a tuple or a list of tuples, where each tuple is is a filter size. If only one size is given, it is applied to all cnn layers. If multiple sizes are given, each size correspond to one layer.
        '''
        super().__init__(name)

        self.output_shape = output_shape
        if isinstance(activation,str):
            self.act = getattr(jax.nn,activation)
        else:
            self.act = activation
        self.dropout_rate = dropout_rate

        if len(cnn_filters[0]) != 2:
            raise ValueError(f'Filters must have length 2, currently received the first filter {cnn_filters[0]}.')
            
        if np.prod(output_shape) != mlp_layers[-1]:
            logger.error(f'Cannot reshape array with {mlp_layers[-1]} elements to shape {output_shape}.')
            raise ValueError('Cannot reshape the output of the MLP to the required output shape.')

        # if only one filter is given, then use the same filter size for all convoluton layers. Otherwise use the provided filter size for individual layer.
        if len(cnn_filters) == 1:
            logger.info('Received only one convolution filter size, use the same size filter for all layers.')
            self.cnn_filters = [cnn_filters[0] for _ in range(len(cnn_channels))]
        elif len(cnn_filters) != len(cnn_channels):
            logger.error(f'Received {len(cnn_filters)} filter size for {len(cnn_channels)} layers. Cannot generate network.')
            raise ValueError('Number of filters do not match the number of layers.')
        else:
            self.cnn_filters = cnn_filters

    
        # define network
        self._mlp = MLP(mlp_layers,activation=self.act,dropout_rate=self.dropout_rate,w_init=w_init,**mlp_kwargs)
        cnn_layers = []
        for i, (c,f) in enumerate(zip(cnn_channels,self.cnn_filters)):
            cnn_layers.append(
                hk.Conv2D(c,f,w_init=w_init,name=f'convolve_{i}',**conv_kwargs)
            )
        self.cnn_layers = tuple(cnn_layers)
    

    def __call__(self,x,training):
        '''Apply network to x. No dropout is training is false. Activation function is not applied after the last layer.'''
        if training:
            logger.info('Model is called in training mode.')
        else:
            logger.info('Model is called in prediction mode.')   
        
        # if x.dims
        # logger.debug(f'Batch size {x.shape[0]} is determined by ')


        out = self._mlp(x,training)
        logger.debug(f'Output of the MLP has shape {out.shape}.')
        out = self.act(out)

        out = jnp.reshape(out,(-1,)+self.output_shape)
        logger.debug(f'Reshaped the output of MLP to {out.shape}.')

        for layer in self.cnn_layers[:-1]:
            out = layer(out)
            # apply dropout
            if training and (self.dropout_rate is not None):
                logger.debug('Doing dropout.')
                out = hk.dropout(hk.next_rng_key(), self.dropout_rate, out)
            out = self.act(out)
        
        # last layer
        out = self.cnn_layers[-1](out)
        
        return out


class Model(BaseModel):
    '''Container for a jax feedforward+cnn model.\n
    
    Has methods:\n
        init: same as haiku.Transformed.init.\n
        apply: same as haiku.Transformed.apply.\n
        predict: apply in prediction mode.
    ''' 
    def __init__(self, 
                mlp_layers:Sequence[int],
                output_shape:Sequence[int],
                cnn_channels:Sequence[int],
                cnn_filters:List[tuple],
                dropout_rate:Optional[float] = None,
                mlp_kwargs:dict = {},
                conv_kwargs:dict = {}) -> None:
        '''Initialise class.
        
        Arguments:\n
            mlp_layers: layers: a list of the number of nodes in each mlp layer.\n
            output_shape: list of dimensions of the target output.\n
            cnn_channels: list of number of output channels, each corresponds to one convolution layer. \n
            cnn_filters: a tuple or a list of tuples, where each tuple is is a filter size. If only one size is given, it is applied to all cnn layers. If multiple sizes are given, each size correspond to one layer.
        '''
        if cnn_channels[-1] != output_shape[-1]:
            warnings.warn('Output shape of the network may not be expected. The output shape of the network does not match the output shape of the mlp layers.')

        super().__init__()

        def forward_fn(x,training=True):
            mdl = MLPWithCNN(mlp_layers,
                            output_shape,
                            cnn_channels,
                            cnn_filters,
                            dropout_rate=dropout_rate,
                            mlp_kwargs=mlp_kwargs,
                            conv_kwargs=conv_kwargs)
            return mdl(x,training)
        
        self.mdl = hk.transform(forward_fn)
        self._apply = jax.jit(self.mdl.apply,static_argnames=['training'])
        self._init = jax.jit(self.mdl.init)

        # every time jax.jit is called, the function recompiles.
        self._predict = jax.jit(jax.tree_util.Partial(self.mdl.apply,training=False))


        logger.info('Successfully created model.')


    def init(self, rng, sample_input) -> hk.Params:
        '''Initialise params'''
        params = self._init(rng,sample_input)
        return params
    
    def apply(self, params:hk.Params, rng:jax.random.PRNGKey, *args, **kwargs):
        '''hk.Transformed.apply, training mode by default\n
        
        Arguments:\n
            params: hk.Params.\n
            rng: jax random number generator key.\n
            Also takes positional and keyword arguments for hk.Transformed.apply.
        '''
        return self._apply(params, rng, *args, **kwargs)
    
    def predict(self,params:hk.Params,x,**kwargs):
        '''Same as apply, but Training flag is False and no randomness.'''
        return self._predict(params,None,x,**kwargs)
