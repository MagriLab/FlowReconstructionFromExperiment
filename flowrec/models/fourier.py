import logging
logger = logging.getLogger(f'fr.{__name__}')
import warnings

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from .feedforward import MLP
from ._general import BaseModel

from typing import Optional, Callable, Sequence, List

class Fourier2Branch(hk.Module):
    raise NotImplementedError



class Model(BaseModel):
    raise NotImplementedError