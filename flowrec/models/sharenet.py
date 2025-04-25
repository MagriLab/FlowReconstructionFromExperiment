import logging
logger = logging.getLogger(f'fr.{__name__}')
import jax
import jax.numpy as jnp
import haiku as hk

from ._general import BaseModel
from .layers import MyConv
from .fourier2branch import Fourier2Branch
from .._typing import *
from ..training_and_states import params_merge

from typing import Callable, Sequence, Tuple
from jax.tree_util import Partial


class ShareNet(hk.Module):
    """3D network, sharing some weights in the homogeneous directions

    Currently only allows one homogeneous direction

    Structure
    -----------------
    - inputs -> input_processing with convolution and linear layers -> Fourier2Branch applied over a homogeneous direction -> 
    """
    def __init__(
        self, 
        img_shapes:NestedTupleInteger,
        b1_channels:Sequence[int],
        b2_channels:Sequence[int],
        b3_channels:Sequence[int],
        img_shapes3d:NestedTupleInteger,
        channels3d:Sequence[int],
        filters3d:Sequence[int],
        padding:str = 'CIRCULAR',
        map_axis:Tuple[int,int] = [2,3], # map which input axis to which output axis 
        activation:Callable[[jnp.ndarray],jnp.ndarray] = jax.nn.tanh,
        small_mlp:bool = False,
        name:str = 'sharenet',
        **kwargs
    ):
        """Initialise with
        -----------------------
        - 
        """
        super().__init__(name)

        if isinstance(activation,str):
            self.act = getattr(jax.nn,activation)
        else:
            self.act = activation
        assert len(map_axis) == 2
        assert len(channels3d) == len(filters3d)
        self.img_shapes3d = img_shapes3d
        logger.debug(f'After applying the shared inner network, the output will be reshaped to {self.img_shapes3d}.')
        logger.info(f'The final output is expected to have shape {[-1,*self.img_shapes3d[-1],channels3d[-1]]}.')

        ## processing inputs
        self.layer_inn1 = MyConv(3,3,padding=padding)
        self.layer_inn2 = hk.Linear(1)

        ## apply the same inner network to all z
        inner = Fourier2Branch(
            img_shapes = img_shapes,
            b1_channels = b1_channels,
            b2_channels = b2_channels,
            b3_channels = b3_channels,
            small_mlp = small_mlp,
            padding = padding,
            activation = self.act,
            **kwargs
        )
        def _over_z(p1, training): #[t,z]
            out1 = inner(p1, training=training) # [t,x,y,c]
            return self.act(out1)
        self.inner_over_z = jax.vmap(_over_z, (map_axis[0],None), map_axis[1])

        ## last layers after shared weigths
        self.last_conv = []
        for c,k in zip(channels3d, filters3d):
            self.last_conv.append(
                MyConv(c, k, padding=padding),
            )

        # define resizing
        if 'resize_method' in kwargs:
            resize_method = kwargs['resize_method']
        else:
            resize_method = 'linear'
        v_resize = jax.vmap(Partial(jax.image.resize,method=resize_method),(-1,None),-1)
        self.vv_resize = jax.vmap(v_resize,(0,None),0)

        
    def __call__(self, x, training):

        if training:
            logger.info('Model is called in training mode.')
        else:
            logger.info('Model is called in prediction mode.')
        input_shape = x.shape
        logger.debug(f'Received input with shape {input_shape}')

        inn = x.reshape(input_shape+(1,))
        out = self.layer_inn1(inn) # [t,x,z,c]
        out = jnp.concatenate((inn,out), axis=-1) # [t,x,z,c']
        out = self.layer_inn2(out) # [t,x,z,1]
        out = out.reshape(input_shape) # [t,x,z]

        ## now apply inner over z
        out = self.inner_over_z(out, training) # [t,x,y,z,c]
        for newshape, layer in zip(self.img_shapes3d, self.last_conv):
            out = self.act(out)
            out = self.vv_resize(out, newshape)
            out = layer(out)
        
        return out



class Model(BaseModel):

    def __init__(
            self,
            img_shapes3d:NestedTupleInteger,
            channels3d:Sequence[int],
            filters3d:Sequence[int]|int,
            img_shapes:NestedTupleInteger = ((32,32), (16,16), (8,8), (16,16), (32,32)),
            b1_channels:Sequence[int] = (1,),
            b2_channels:Sequence[int] = (8,16,8),
            b3_channels:Sequence[int] = (4,),
            map_axis:Tuple[int,int] = (2,3),
            padding:str = 'CIRCULAR',
            name:str = 'sharenet',
            **kwargs
    ):
        if name:
            self.name = name # name is needed for loading from pre-trained
        else:
            self.name = 'sharenet'

        if isinstance(filters3d, int):
            filters3d = [filters3d,]*len(channels3d)

        def forward_fn(x, training=True):
            mdl = ShareNet(
                img_shapes = img_shapes,
                b1_channels = b1_channels,
                b2_channels = b2_channels,
                b3_channels = b3_channels,
                img_shapes3d = img_shapes3d,
                channels3d = channels3d,
                filters3d = filters3d,
                padding = padding,
                map_axis = map_axis,
                name = name,
                **kwargs
            )
            return mdl(x, training)
        self.mdl = hk.transform(forward_fn)
        logger.debug('Successfully created model, passing the model to BaseModel.')
        super().__init__(self.mdl)

    @staticmethod
    def count_params(*params):
        """Print the number of parmeters. If multiple is provided then print the total sum."""
        all_params = params_merge(*params)
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(all_params))
        print(f'Number of parameters {param_count:,}')