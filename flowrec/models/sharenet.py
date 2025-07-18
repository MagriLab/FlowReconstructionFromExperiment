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
from ..physics_and_derivatives import derivative1

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

        ## handle the extra inputs
        self.divfree = False
        if 'divfree' in kwargs:
            self.divfree = kwargs.pop('divfree')
            if self.divfree:
                logger.warning('Forcing divergence free output.')

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
        
        if self.divfree:
            out_u = _cross_product(out[...,:-1])
            out_p = out[...,-1]
            out = jnp.concatenate([out_u, out_p[:,:,:,:,None]], axis=-1)
        
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



def _cross_product(u: Array):
    # u: [t, x, ..., u]
    # axis_space = list(range(1,v.ndim-1))
    u = jnp.moveaxis(u[...],-1,0) # move velocity axis to 0
    v_derivative1 = jax.vmap(derivative1,(0,None,None),0)
    def _didj(de_fun,inn):
        didj_T = de_fun(inn,1.0,1).reshape((-1,)+inn.shape)
        for j in range(1,u.shape[0]):
            didj_T = jnp.concatenate(
                (
                didj_T,
                de_fun(inn,1.0,j+1).reshape((-1,)+inn.shape)
                ),
                axis=0
            )
        return didj_T # for de_fun = v_derivative1 and inn=u -> [j,i,t,x,y,z]
    dui_dxj = jnp.einsum('jit... -> ijt...', _didj(v_derivative1, u)) # [i,j,t,x,y,z]
    cross0 = dui_dxj[2,1,...] - dui_dxj[1,2,...]
    cross1 = dui_dxj[0,2,...] - dui_dxj[2,0,...]
    cross2 = dui_dxj[1,0,...] - dui_dxj[0,1,...]
    cross = jnp.stack([cross0,cross1,cross2])
    return jnp.einsum('utxyz -> txyzu', cross)