import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .models._general import BaseModel

import chex
from typing import Union, Optional

Array = Union[np.ndarray, jnp.ndarray]
Model = Union[BaseModel, hk.Transformed]


def loss_mse(model:Model, params:hk.Params, x:Array, y:Array):
    '''Mean squared error for use as loss in training.'''
    pred = model.apply(params,x)
    loss = mse(pred,y)
    return loss


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