import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .models._general import BaseModel

import chex
from typing import Union, Optional, Callable
import logging
logger = logging.getLogger(f'fr.{__name__}')

Array = Union[np.ndarray, jnp.ndarray]
Model = Union[BaseModel, hk.Transformed]


# ====================== for during training =========================

def loss_mse(apply_fn:Callable, 
            params:hk.Params, 
            rng:jax.random.PRNGKey, 
            x:Array, 
            y:Array,
            **kwargs) -> float:
    '''Mean squared error for use as loss in training.
    
    Arguments:\n
        apply_fn: an hk.Transformed.apply that takes (params, rng, inputs) and returns an output.\n
        params: params for apply_fn.\n
        rng: jax.random.PRNGKey for apply_fn.\n
        x: input.\n
        y: ground truth.\n
        *args, **kwargs: for apply_fn.\n
    
    Returns:\n
        loss: the mean squared error between y and predicted y (from x).
    '''
    pred = apply_fn(params, rng, x, **kwargs)
    loss = mse(pred,y)
    return loss



# ====================== regular calculations ========================

def mse(pred:Array,true:Array) -> float:
    '''Mean squared error
    
    Argument:\n
        pred: predicted array.\n
        true: ground truth array.\n

    Return:
        loss: (1/n)(pred-true)**2
    '''
    if true is not None:
        # Avoid broadcasting logic for "-" operator.
        chex.assert_equal_shape((pred, true))
        loss = jnp.mean((pred-true)**2)
    else:
        loss = jnp.mean(pred**2)
    
    return loss


def sum_squared_element(tree):
    '''Sum of squared of elements of all leaves in a pytree (e.g. a hk.Params).'''
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    leaves = jax.tree_util.all_leaves(squared)
    out = np.sum(leaves)
    return out