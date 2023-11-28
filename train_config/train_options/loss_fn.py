from flowrec._typing import *
from ml_collections.config_dict import ConfigDict

from flowrec.utils.py_helper import slice_from_tuple
from flowrec.data import data_partition, unnormalise_group, normalise
from flowrec import losses
from flowrec import physics_and_derivatives as derivatives

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from typing import Callable, Sequence
from haiku import Params

import logging
logger = logging.getLogger(f'fr.{__name__}')


def loss_fn_example(config:ConfigDict, **kwargs):
    '''# Example function that builds a loss function for training.\n
    Pass 'loss_fn@example' to the config to use this function in training.\n
    Takes the config ('cfg' in train.py) and returns a (must be jitable) loss function for use in training.

    The loss function returned must take the following list of arguments in this order:\n
    1. apply_fn: a haiku Transformed apply function
    2. params: haiku.Params
    3. rng: jax.random.PRNGKey
    4. x: input to the network, it will be used in apply_fn
    5. y: labels
    6. Any other positional arguments
    6. apply_kwargs: dictionary containing the kwargs to apply_fn
    7. **kwargs
    '''

    # available kwargs:
    # take_observation:callable
    # insert_observation:callable
    # datainfo:MetadataTree

    def loss_fn(apply_fn:Callable,
                params:Params, 
                rng:jax.random.PRNGKey, 
                x:Sequence[jax.Array], 
                y:Sequence[jax.Array], 
                apply_kwargs:dict = {},
                **kwargs):
        
        # available kwargs:
        # normalise:bool
        # y_minmax:jax.Array

        # Required elements with example values
        loss_div = 1
        loss_mom = 1
        loss_sensor = 1
        
        loss = loss_div + loss_mom + loss_sensor

        return loss, (loss_div, loss_mom, loss_sensor)
        
    return loss_fn



def loss_fn_physicswithdata(cfg,**kwargs):
    '''Insert observations into prediction, then train on physics loss (or physics+sensor loss).'''

    take_observation: Callable = kwargs['take_observation']
    insert_observation: Callable = kwargs['insert_observation']
    
    datainfo = kwargs['datainfo']

    wdiv= cfg.train_config.weight_continuity
    wmom = cfg.train_config.weight_momentum
    ws = cfg.train_config.weight_sensors

    if ws > 0.0:
        logger.info('Training with both physics and sensor loss even though sensor measurements have already been inserted into the prediction.')
    
    f = _is_forced(**kwargs)

    def loss_fn(apply_fn:Callable,
                params:Params, 
                rng:jax.random.PRNGKey, 
                x:Sequence[jax.Array], 
                y:Sequence[jax.Array], 
                normalise:bool, 
                y_minmax:jax.Array = jnp.array([]),
                apply_kwargs:dict = {}, 
                **kwargs):
        pred = apply_fn(params, rng, x, **apply_kwargs)
        pred_observed = take_observation(pred)
        loss_sensor = losses.mse(pred_observed, y)

        pred_new = insert_observation(pred,y)

        # normalise
        if normalise:
            pred_new = unnormalise_group(pred_new, y_minmax, axis_data=-1, axis_range=0)
            logger.debug('Un-normalise before calculating loss.')

        loss_div = losses.divergence(pred_new[...,:-1], datainfo)
        loss_mom = losses.momentum_loss(
            u=pred_new,
            datainfo=datainfo,
            forcing=f
        )
        
        
        return wdiv*loss_div+wmom*loss_mom+ws*loss_sensor, (loss_div,loss_mom,loss_sensor)
    
    return Partial(loss_fn, normalise=cfg.data_config.normalise)


def loss_fn_physicsnoreplace(cfg,**kwargs):
    '''Train on physics loss + sensor loss.'''

    take_observation:Callable = kwargs['take_observation']
    datainfo = kwargs['datainfo']

    wdiv= cfg.train_config.weight_continuity
    wmom = cfg.train_config.weight_momentum
    ws = cfg.train_config.weight_sensors

    f = _is_forced(**kwargs)

    def loss_fn(apply_fn:Callable,
                params:Params, 
                rng:jax.random.PRNGKey, 
                x:Sequence[jax.Array], 
                y:Sequence[jax.Array],
                normalise:bool, 
                y_minmax:jax.Array = jnp.array([]),
                apply_kwargs:dict = {}, 
                **kwargs):
        pred = apply_fn(params, rng, x, **apply_kwargs)
        pred_observed = take_observation(pred)
        loss_sensor = losses.mse(pred_observed, y)

        # normalise
        if normalise:
            pred = unnormalise_group(pred, y_minmax, axis_data=-1, axis_range=0)
            logger.debug('Un-normalise before calculating loss.')

        loss_div = losses.divergence(pred[...,:-1], datainfo)
        loss_mom = losses.momentum_loss(
            u=pred,
            datainfo=datainfo,
            forcing=f
        )
        
        return wdiv*loss_div+wmom*loss_mom+ws*loss_sensor, (loss_div,loss_mom,loss_sensor)
    
    return Partial(loss_fn, normalise=cfg.data_config.normalise)



def loss_fn_physicsreplacemean(cfg,**kwargs):
    '''Train on physics loss calculated from pred_new, where \n
    pred_new = pred at hidden locations and \n
    pred_new = pred - mean(pred) + mean(observed), mean() is averaging in time.
    '''

    take_observation: Callable = kwargs['take_observation']
    insert_observation: Callable = kwargs['insert_observation']
    
    datainfo = kwargs['datainfo']

    wdiv= cfg.train_config.weight_continuity
    wmom = cfg.train_config.weight_momentum
    ws = cfg.train_config.weight_sensors

    f = _is_forced(**kwargs)

    def loss_fn(apply_fn:Callable,
                params:Params, 
                rng:jax.random.PRNGKey, 
                x:Sequence[jax.Array], 
                y:Sequence[jax.Array],
                normalise:bool, 
                y_minmax:jax.Array = jnp.array([]),
                apply_kwargs:dict = {}, 
                **kwargs):
        pred = apply_fn(params, rng, x, **apply_kwargs)
        pred_observed = take_observation(pred)
        loss_sensor = losses.mse(pred_observed, y)
        
        pred_replaced = insert_observation(pred,y)
        pred_f = pred - jnp.mean(pred,axis=0,keepdims=True)
        pred_new = pred_f + jnp.mean(pred_replaced,axis=0,keepdims=True)
        
        # normalise
        if normalise:
            pred_new = unnormalise_group(pred_new, y_minmax, axis_data=-1, axis_range=0)
            logger.debug('Un-normalise before calculating loss.')
        
        loss_div = losses.divergence(pred_new[...,:-1], datainfo)
        loss_mom = losses.momentum_loss(
            u = pred_new,
            datainfo = datainfo,
            forcing = f
        )
        
        return wdiv*loss_div+wmom*loss_mom+ws*loss_sensor, (loss_div,loss_mom,loss_sensor)


    return Partial(loss_fn, normalise=cfg.data_config.normalise)



def loss_fn_physicsandmean(cfg, **kwargs):
    '''Train on physics loss + sensor loss, \n
    where the sensor loss is the mse(mean(pred), mean(observed)) at observed locations.\n
    mean() is time averaging'''

    take_observation:Callable = kwargs['take_observation']
    datainfo = kwargs['datainfo']

    wdiv= cfg.train_config.weight_continuity
    wmom = cfg.train_config.weight_momentum
    ws = cfg.train_config.weight_sensors
    
    logger.warn("Are you looking for 'physicsreplacemean'")

    f = _is_forced(**kwargs)

    def loss_fn(apply_fn:Callable,
                params:Params, 
                rng:jax.random.PRNGKey, 
                x:Sequence[jax.Array], 
                y:Sequence[jax.Array],
                normalise:bool, 
                y_minmax:jax.Array = jnp.array([]),
                apply_kwargs:dict = {}, 
                **kwargs):
        pred = apply_fn(params, rng, x, **apply_kwargs)
        pred_observed = take_observation(pred)
        
        ## mean loss
        loss_sensor = losses.mse(jnp.mean(pred_observed,axis=0), jnp.mean(y,axis=0))

        # normalise
        if normalise:
            pred = unnormalise_group(pred, y_minmax, axis_data=-1, axis_range=0)
            logger.debug('Un-normalise before calculating loss.')

        loss_div = losses.divergence(pred[...,:-1], datainfo)
        loss_mom = losses.momentum_loss(
            u=pred,
            datainfo=datainfo,
            forcing=f
        )
        
        return wdiv*loss_div+wmom*loss_mom+ws*loss_sensor, (loss_div,loss_mom,loss_sensor)

    return Partial(loss_fn, normalise=cfg.data_config.normalise)


def _is_forced(**kwargs):
    if 'forcing' in kwargs:
        forcing = kwargs['forcing']
        logger.debug('Forced flow.')
    else:
        forcing = None
    return forcing