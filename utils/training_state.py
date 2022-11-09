import jax
import haiku as hk
import optax
import numpy as np
import pickle
from pathlib import Path
from typing import Union, NamedTuple

# taken from https://github.com/deepmind/dm-haiku/issues/18
def save(ckpt_dir: Union[Path,str], 
            state,
            f_name:str) -> None:
    with open(Path(ckpt_dir, (f_name + ".npy")), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_util.tree_map(lambda _: 0, state)
    with open(Path(ckpt_dir, (f_name + ".pkl")), "wb") as f:
        pickle.dump(tree_struct, f)

def restore(ckpt_dir, f_name:str):
    with open(Path(ckpt_dir, (f_name + ".pkl")), "rb") as f:
        tree_struct = pickle.load(f)
 
    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(Path(ckpt_dir, (f_name + ".npy")), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, flat_state)


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState