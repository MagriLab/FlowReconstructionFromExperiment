import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .models._general import BaseModel
from .physics_and_derivatives import div_field, momentum_residue_field

import chex
from typing import Union, Optional, Callable
import logging
logger = logging.getLogger(f'fr.{__name__}')

Array = Union[np.ndarray, jnp.ndarray]
Model = Union[BaseModel, hk.Transformed]
Scalar = Union[int,float]


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
    out = jnp.sum(leaves)
    return out


def divergence(
    ux:Array,
    uy:Array,
    dx:Scalar,
    dy:Scalar,
    x_axis:int,
    y_axis:int,
    uz:Optional[Array]=None,
    dz:Optional[Scalar]=None,
    z_axis:Optional[int]=None)-> float:

    '''Calculate the diveregence loss.
    The diveregence loss the the squared mean of the divergence field.
    
    Arguments:\n
        ux: an array of velocity in x direction.\n
        uy: an array of velocity in x direction.\n
        dx: space in the x direction between each value in ux.\n
        dy: space in the y direction between each value in uy.\n
        x_axis: which axis is x axis.\n
        y_axis: which axis is y axis.\n
        uz: an array of velocity in z direction, None if the 2D flow.\n
        dz: space in the z direction between each value in uz, None if the 2D flow.\n
        z_axis: which axis is z axis, None if the 2D flow.\n

    Return:\n
        A non-negative scalar, square of the mean of the divergence field.
    '''

    div_field = div_field(ux,uy,dx,dy,x_axis,y_axis,uz=uz,dz=dz,z_axis=z_axis)
    div_loss = jnp.mean(div_field**2) # div_loss >= 0

    return div_loss


#         # Use a dataclass for ux, uy, uz, p, index, dt, dx
# def physics_residue(
#         ux:Array,
#         uy:Array,




# ) -> float: