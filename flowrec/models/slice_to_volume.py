import logging
logger = logging.getLogger(f'fr.{__name__}')
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from ._general import BaseModel
from .._typing import *

from typing import Optional, Callable, Sequence, List, Tuple
from jax.tree_util import Partial



class Slice3D(hk.Module):
    '''2D network trained on 3D data.'''

    def __init__(
            self,
            name:Optional[str] = None,
    ):
        super().__init__(name=name)


    def __call__(self, x, training):
        pass


class Model(BaseModel):
    '''
    Has methods:\n
        init: same as haiku.Transformed.init.\n
        apply: same as haiku.Transformed.apply.\n
        predict: apply in prediction mode.
    ''' 
