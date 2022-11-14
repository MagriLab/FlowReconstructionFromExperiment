import logging
logger = logging.getLogger(f'fr.{__name__}')

from ._general import BaseModel

import jax 
import haiku as hk


class Model(BaseModel):
    def __init__(self, 
                layers:list, 
                rng:jax.random.PRNGKey,
                APPLY_RNG:bool = True,
                activation=jax.nn.tanh,
                w_init=hk.initializers.VarianceScaling(1.0,"fan_avg","uniform"),
                **kwargs) -> None:
        '''Make a feedforward model.

        Arguments:\n
            layers: a list of the number of nodes in each layer.\n
            rng: a jax 
        '''
        super().__init__()
        self.layers = layers
        self.rng = rng
        self.APPLY_RNG = APPLY_RNG

        # define network
        def feedforward(x):
            mlp = hk.nets.MLP(self.layers,
                                activation=activation,
                                w_init=w_init,
                                **kwargs)
            return mlp(x)
        self.mdl = hk.transform(feedforward)
        if not APPLY_RNG:
            logger.info("Use without_apply_rng when making a feedforward network.")
            self.mdl = hk.without_apply_rng(self.mdl)

        self._apply = jax.jit(self.mdl.apply)
        self._init = jax.jit(self.mdl.init)
        logger.info(f'Successfully created a MLP.')
    
    def init(self, sample_input) -> hk.Params:
        '''Initialise params'''
        params = self._init(self.rng, sample_input)
        return params

    def apply(self, params: hk.Params, *args, **kwargs):
        '''hk.Transformed.apply
        
        Arguments:\n
            params: hk.Params.\n
            Also takes positional and keyword arguments for hk.Transformed.apply.
        '''
        if self.APPLY_RNG:
            return self._apply(params, self.rng, *args, **kwargs)
        else:
            return self._apply(params, *args, **kwargs)