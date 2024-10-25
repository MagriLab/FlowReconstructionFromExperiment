import logging
logger = logging.getLogger(f'fr.{__name__}')
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from .fourier2branch import Fourier2Branch
from ._general import BaseModel
from .._typing import *

from typing import Optional, Callable, Sequence, List, Tuple
from jax.tree_util import Partial


raise NotImplementedError