import logging
logger = logging.getLogger(f'fr.{__name__}')

import jax
from jax.tree_util import Partial
import haiku as hk
import optax
import numpy as np
import pickle
from pathlib import Path


from .models._general import BaseModel

from typing import Union, NamedTuple, Callable, Any

Array = Union[np.ndarray, jax.numpy.ndarray]
Model = Union[BaseModel, hk.Transformed]

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


def generate_update_fn(apply_fn:Callable, 
            optax_optimiser, 
            loss_fn:Callable,
            kwargs_loss:dict={},
            kwargs_value_and_grad:dict={}) -> Callable:
    '''Generates a model specifiec function to calculate the weights and update the weights once.
    
    Arguments:\n
        apply_fn: a hk.Transformed.apply function that takes (params,rng,inputs).\n
        optax_optimiser: an optax optimiser.\n
        loss_fn: a loss function that takes (Model,params,input,output). \n
        kwargs_loss: **kwargs for loss_fn

    Return:
        update function
    '''
    loss_fn_partial = jax.jit(Partial(loss_fn,apply_fn,**kwargs_loss))
    logger.debug('Constructing a partial loss function by holding the apply_fn constant.')

    def step(state: TrainingState, rng:jax.random.PRNGKey, x:Array, y:Array):
        '''Update the state once.

        Arguments:\n
            state: a state (class TraningState) containing the params and the optimiser state.\n
            rng: jax random number generator.\n
            x: inputs to the network.\n
            y: target output of the network.\n

        return:\n
            loss: training loss for the current step.\n
            state: updated state.
        '''
        logger.debug(f'Is update functin in jit: {isinstance(jax.numpy.array(0),jax.core.Tracer)}')
        loss,grads = jax.value_and_grad(loss_fn_partial,**kwargs_value_and_grad)(state.params,rng,x,y)
        logger.debug('Successfully calculated loss and taken gradient from partial loss function.')

        updates, opt_state = optax_optimiser.update(grads,state.opt_state,state.params)
        params = optax.apply_updates(state.params,updates)
        logger.debug('Successfully updated weights once.')
        return loss, TrainingState(params,opt_state)
        
    return jax.jit(step)



# taken from https://github.com/deepmind/dm-haiku/issues/18
def save_trainingstate(ckpt_dir:Union[Path,str], 
                        state:TrainingState,
                        f_name:str) -> None:
    '''Save the training state to file.
    
    Arguments:\n
        ckpt_dir: path to directory where the state would be saved.\n
        state: class TrainingState.\n
        f_name: file name to be used (the prefix).\n
    '''
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        logger.warning('Taget directory for saving checkpoint does not exist. Creating Target directory and its parents.')
        ckpt_dir.mkdir(parents=True)

    with open(Path(ckpt_dir, (f_name + ".npy")), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_util.tree_map(lambda _: 0, state)
    with open(Path(ckpt_dir, (f_name + ".pkl")), "wb") as f:
        pickle.dump(tree_struct, f)



def restore_trainingstate(ckpt_dir:Union[Path,str], f_name:str):
    '''Restore the training state from files.
    
    Arguments:
        ckpt_dir: directory where the files are located.\n
        f_name: prefix of the files to use.
    
    returns:
        a training state
    '''
    if not Path(ckpt_dir).is_dir():
        raise ValueError('Cannot find checkpoint directory.')

    with open(Path(ckpt_dir, (f_name + ".pkl")), "rb") as f:
        tree_struct = pickle.load(f)
 
    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(Path(ckpt_dir, (f_name + ".npy")), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, flat_state)

