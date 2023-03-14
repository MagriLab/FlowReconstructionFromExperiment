from typing import Union

import numpy as np
import jax
import haiku as hk

from .models._general import BaseModel
from .data import DataMetadata, _Metadata2d, _Metadata3d

Array = Union[np.ndarray, jax.Array]
Model = Union[BaseModel, hk.Transformed]
Scalar = Union[int, float]
ClassDataMetadata = Union[DataMetadata, _Metadata2d, _Metadata3d]
MetadataTree = Union[_Metadata3d,_Metadata2d]