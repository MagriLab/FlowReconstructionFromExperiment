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
        dropout_rate: must be a floating point number between [0.,1.]. Dropout is not used if dropout_rate is None, or is the network is called with training=False.\n
        name: name of the network.\n
    
    After transforming, has method apply(params,rng,inputs,training=).\n
    '''
    def __init__(self, 
                mlp_layers:list, 
                activation:str|Callable[[jnp.ndarray],jnp.ndarray]='tanh',
                w_init:Optional[hk.initializers.Initializer]=None,
                b_init:Optional[hk.initializers.Initializer]=None,
                has_bias:bool=True,
                dropout_rate:Optional[float] = None,
                name:Optional[str] = None):
        super().__init__(name=name)
        self.w_init = w_init
        self.b_init = b_init
        if isinstance(activation,str):
            self.act = getattr(jax.nn,activation)
        else:
            self.act = activation
        self.has_bias = has_bias
        logger.info('MLP has no bias.') if not has_bias else None
        
        # add layers
        layers = []
        mlp_layers = tuple(mlp_layers)
        for index, output_size in enumerate(mlp_layers):
            layers.append(
                hk.Linear(output_size=output_size,
                            w_init=w_init,
                            b_init=b_init,
                            with_bias = has_bias,
                            name=f'linear_{index}')
            )
        self.layers = tuple(layers)
        self.output_size = mlp_layers[-1] if mlp_layers else None

        self.dropout_rate = dropout_rate
        

    def __call__(self,x,training):
        '''Apply network to inputs x. No dropout if training is false.'''
        if training:
            logger.info('MLP is called in training mode.')
        else:
            logger.info('MLP is called in prediction mode.')   

        num_layers = len(self.layers)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # dropout and activation is not applied after the last layer.
            if i < (num_layers - 1):                
                if training and (self.dropout_rate is not None):
                    logger.debug('Performing dropout.')
                    out = hk.dropout(hk.next_rng_key(), self.dropout_rate, out)
                out = self.act(out)

        return out



class Model(BaseModel):
    '''Container for a jax feedforward model.\n
    
    Has methods:\n
        init: same as haiku.Transformed.init.\n
        apply: same as haiku.Transformed.apply.\n
        predict: apply in prediction mode.
    ''' 

    def __init__(self, 
                mlp_layers:list, 
                activation:str|Callable[[jnp.ndarray],jnp.ndarray]=jax.nn.tanh,
                w_init:Optional[hk.initializers.Initializer]=hk.initializers.VarianceScaling(1.0,"fan_avg","uniform"),
                dropout_rate:Optional[float]=None,
                **mlp_kwargs) -> None:
        '''Initialise class.

        Arguments:\n
            layers: a list of the number of nodes in each layer.\n
            activation: activation function to use in the intermediate layers. Default tanh.\n
            w_init: weight initialisation scheme. Default Golort uniform. \n
            dropout_rate: a float between 0.0 and 1.0. If None, do not use dropout.\n
            mlp_kwargs: keyword arguments to be passed to haiku.nets.MLP.
        '''
        super().__init__()
        self.layers = mlp_layers


        def forward_fn(x,training=True):
            mlp = MLP(self.layers,
                        w_init=w_init,
                        activation=activation,
                        dropout_rate=dropout_rate,
                        **mlp_kwargs
                        )
            return mlp(x,training=training)
        self.mdl = hk.transform(forward_fn)

        self._apply = jax.jit(self.mdl.apply,static_argnames=['training'])
        self._init = jax.jit(self.mdl.init)

        self._predict = jax.jit(jax.tree_util.Partial(self.mdl.apply,training=False))

        logger.info(f'Successfully created a MLP.')
    
    def init(self, rng, sample_input) -> hk.Params:
        '''Initialise params'''
        params = self._init(rng, sample_input)
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
        return self._predict(params,None,x,**kwargs)