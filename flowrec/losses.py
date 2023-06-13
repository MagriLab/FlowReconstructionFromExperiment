import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .models._general import BaseModel
from .physics_and_derivatives import div_field, momentum_residual_field
from .data import DataMetadata

import chex
from typing import Union, Optional, Callable
from ._typing import Array, ClassDataMetadata
import logging
logger = logging.getLogger(f'fr.{__name__}')



# ====================== for during training =========================

def loss_mse(apply_fn:Callable, 
            params:hk.Params, 
            rng:jax.random.PRNGKey, 
            x:Array, 
            y:Array,
            apply_kwargs:dict,
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
    pred = apply_fn(params, rng, x, **apply_kwargs)
    loss = mse(pred,y)
    return loss







# ====================== regular calculations ========================

def mse(pred:Array,true:Optional[Array] = None) -> float:
    '''Mean squared error
    
    Argument:\n
        pred: predicted array.\n
        true: ground truth array.\n

    Return:
        loss: (1/n)(pred-true)**2
    '''
    if true is not None:
        # Avoid broadcasting logic for "-" operator.
        try:
            chex.assert_equal_shape((pred, true))
        except AssertionError as err:
            logger.error('Cannot calculate mean squared error, input shape mismatch.')
            raise err
        loss = jnp.mean((pred-true)**2)
    else:
        loss = jnp.mean(pred**2)
    
    return loss


def sum_squared_element(tree):
    '''Sum of squared of elements of all leaves in a pytree (e.g. a hk.Params).'''
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    leaves = jax.tree_util.all_leaves(squared)
    out = jnp.sum(leaves)
    return out


def divergence(
    u:Array,
    datainfo:ClassDataMetadata) -> float:

    '''Calculate the diveregence loss.
    The diveregence loss the the squared mean of the divergence field.
    
    Arguments:\n
        u: an array of velocities with shape [t,x,y,...,i], i=2 if 2D flow, 3 if 3D flow. \n
        datainfo: an instance of DataMetadata.\n

    Return:\n
        A non-negative scalar, square of the mean of the divergence field.
    '''

    div = div_field(u,datainfo)
    div_loss = jnp.mean(div**2) # div_loss >= 0

    return div_loss


def relative_error(pred,true):
    err = np.sqrt(
        np.einsum('t x y -> ', (pred-true)**2)
        / np.einsum('t x y -> ', true**2)
    )
    return err