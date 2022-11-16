import logging
logger = logging.getLogger(f'fr.{__name__}')

from ._general import BaseModel

import jax 
import jax.numpy as jnp
import haiku as hk

from typing import Optional, Callable



class MLP(hk.Module):
    '''A simple MLP.\n
    Mostly the same as haiku.nets.MLP, but with more user-friendly dropout.\n

    Initialise with:\n
        output_sizes: list of output size of all layers [layer1, layer2, ...].\n
        w_init: Initializer for weights.\n
        activation: activation function to apply between layers (not applied after the last layer).\n
        b_init: Initializer for bias.\n
        HAS_BIAS: if False, do not use bias. Default true.\n
        dropout_rate: must be a floating point number between [0.,1.]. Dropout is not used if dropout_rate is None, or is the network is called with TRAINING=False.\n
        name: name of the network.\n
    
    After transforming, has method apply(params,rng,inputs,TRAINING=).\n
    '''
    def __init__(self, 
                output_sizes:list, 
                w_init:Optional[hk.initializers.Initializer]=None,
                activation:Callable[[jnp.ndarray],jnp.ndarray]=None,
                b_init:Optional[hk.initializers.Initializer]=None,
                HAS_BIAS:bool=True,
                dropout_rate:Optional[float] = None,
                name:Optional[str] = None):
        super().__init__(name=name)
        self.w_init = w_init
        self.b_init = b_init
        self.act = activation
        self.HAS_BIAS = HAS_BIAS
        logger.info('MLP has no bias.') if not HAS_BIAS else None
        
        # add layers
        layers = []
        output_sizes = tuple(output_sizes)
        for index, output_size in enumerate(output_sizes):
            layers.append(
                hk.Linear(output_size=output_size,
                            w_init=w_init,
                            b_init=b_init,
                            with_bias = HAS_BIAS,
                            name=f'linear_{index}')
            )
        self.layers = tuple(layers)
        self.output_size = output_sizes[-1] if output_sizes else None

        self.dropout_rate = dropout_rate
        

    def __call__(self,x,TRAINING):
        '''Apply network to inputs x. No dropout if TRAINING is false.'''
        if TRAINING:
            logger.debug('MLP is called in training mode.')
        else:
            logger.debug('MLP is called in prediction mode.')   

        num_layers = len(self.layers)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # dropout and activation is not applied after the last layer.
            if i < (num_layers - 1):                
                if TRAINING and (self.dropout_rate is not None):
                    out = hk.dropout(hk.next_rng_key(), self.dropout_rate, out)
                out = self.act(out)

        return out



class Model(BaseModel):
    def __init__(self, 
                layers:list, 
                rng:jax.random.PRNGKey,
                activation:Callable[[jnp.ndarray],jnp.ndarray]=jax.nn.tanh,
                w_init:Optional[hk.initializers.Initializer]=hk.initializers.VarianceScaling(1.0,"fan_avg","uniform"),
                dropout_rate:Optional[float]=None,
                **mlp_kwargs) -> None:
        '''Container for a jax feedforward model. 

        Arguments:\n
            layers: a list of the number of nodes in each layer.\n
            rng: a jax random number generator.\n
            APPLY_RNG: If False, the .appply() method does not use rng.\n
            activation: activation function to use in the intermediate layers. Default tanh.\n
            w_init: weight initialisation scheme. Default Golort uniform. 
            mlp_kwargs: keyword arguments to be passed to haiku.nets.MLP.
        '''
        super().__init__()
        self.layers = layers
        self.rng = rng


        def forward_fn(x,TRAINING=True):
            mlp = MLP(self.layers,
                        w_init=w_init,
                        activation=activation,
                        dropout_rate=dropout_rate,
                        **mlp_kwargs
                        )
            return mlp(x,TRAINING=TRAINING)
        self.mdl = hk.transform(forward_fn)

        self._apply = jax.jit(self.mdl.apply,static_argnames=['TRAINING'])
        self._init = jax.jit(self.mdl.init)

        logger.info(f'Successfully created a MLP.')
    
    def init(self, sample_input) -> hk.Params:
        '''Initialise params'''
        params = self._init(self.rng, sample_input)
        return params

    def apply(self, params: hk.Params, rng:jax.random.PRNGKey, *args, **kwargs):
        '''hk.Transformed.apply, training mode by default\n
        
        Arguments:\n
            params: hk.Params.\n
            rng: jax random number generator key.\n
            Also takes positional and keyword arguments for hk.Transformed.apply.
        '''
        return self._apply(params, rng, *args, **kwargs)

    def predict(self, params:hk.Params, x, **kwargs):
        '''Same as apply, but Training flag is False and no randomness.'''
        return self._apply(params,None,x,TRAINING=False,**kwargs)