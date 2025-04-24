import jax
import warnings
import haiku as hk
import logging
logger = logging.getLogger(f'fr.{__name__}')
from functools import partial

class BaseModel():
    """Has Methods
    ------------------
        - init: same as haiku.Transformed.init.
        - apply: same as haiku.Transformed.apply.
        - predict: apply in prediction mode.
        - set_nontrainable: make the model remember the non-trainable weights
        - apply_trainable: apply the model using the trianable weights and the previously memorised non-trainable weights.
    """
    def __init__(self, mdl:hk.Transformed|None = None) -> None:
        if mdl is not None:
            logger.debug('Creating model in ')
            self.mdl = mdl
            self._apply = jax.jit(self.mdl.apply,static_argnames=['training'])
            self._init = jax.jit(self.mdl.init)
            self._predict = jax.jit(jax.tree_util.Partial(self.mdl.apply,training=False))
            logger.info('Successfully created model.')
        else:
            logger.warning('BaseModel did not receive a model. The user must override all methods.')

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
    
    def predict(self,params:hk.Params,x,**kwargs):
        '''Same as apply, but Training flag is False and no randomness.'''
        return self._predict(params,None,x,**kwargs)

    def set_nontrainable(self, non_trainable_params:hk.Params):
        '''Make the model remeber the non_trainable_params'''
        if hasattr(self, 'params_merge'):
            warnings.warn('Overriding the old non_trainable_params.')
        self.params_merge = partial(hk.data_structures.merge, non_trainable_params)

    def apply_trainable(self, trainable_params:hk.Params, rng:jax.random.PRNGKey, *args, **kwargs):
        '''Apply the model using the trainable_params.
        ------------------------------
        Must call `set_nontrainable(non_trainable_params)` first.
        '''
        params = self.params_merge(trainable_params)
        return self._apply(params, rng, *args, **kwargs)
