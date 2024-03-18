import logging
logger = logging.getLogger(f'fr.{__name__}')
import warnings

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from .feedforward import MLP
from ._general import BaseModel
from .._typing import *

from typing import Optional, Callable, Sequence, List, Tuple
from jax.tree_util import Partial


def _empty_fun(x):
    return x


def _test_cnn_filters(cnn_filters,cnn_channels):
    """Make cnn filters from user input. If only one filter size is given, then the same filter size will be applied to all layers."""
    if len(cnn_filters) == 1:
        logger.info('Received only one convolution filter size, use the same size filter for all layers.')
        return [cnn_filters[0] for _ in range(len(cnn_channels))]
    elif len(cnn_filters) != len(cnn_channels):
        logger.error(f'Received {len(cnn_filters)} filter size for {len(cnn_channels)} layers. Cannot generate network.')
        raise ValueError('Number of filters do not match the number of layers.')
    else:
        return cnn_filters



class Fourier2Branch(hk.Module):
    """A two branch model, branch 1 is fourier convolution, branch 2 is a normal convolution.



    Attributes
    --------------------
    - b0_shape: tuple, the shape of the image in branch 0. (x1,x2) if the output is 2D, (x1,x2,x3) if the output is 3D.
    - b1_shape: tuple, the shape of the image in branch 1 (Fourier branch).
    - b2_shapes: a sequence of tuples, the shape of the image at each layer in branch 2.
    - output_shape: tuple, the shape of the output. (x1,x2) if the output is 2D, the number of variables are not included.

    Graph of branches
    --------------------
    - inputs --> branch 0 (b0) --> split into branch 1 (b3) / branch 2 (b2) --> merge to branch 3(b3) --> output
    - Branch 0: inputs -> mlp -> reshape into b0_shape -> conv -> b1/b2
    - Branch 1: b0 -> resize to b1_shape -> FFT(unless specified) -> conv ... -> inverse FFT -> b3
    - Branch 2: b0 -> (resize -> conv) ... -> b3
    - Branch 3: concatenate(b1,b2) -> resize -> conv ... -> output


    """
    def __init__(
            self,
            img_shapes:NestedTupleInteger,
            b1_channels:Sequence[int],
            b2_channels:Sequence[int],
            b3_channels:Sequence[int],
            b1_filters:List[Tuple[int, ...]] = [(3,3),],
            b2_filters:List[Tuple[int, ...]] = [(3,3),],
            b3_filters:List[Tuple[int, ...]] = [(3,3),],
            activation:Callable[[jnp.ndarray],jnp.ndarray] = jax.nn.tanh,
            w_init:Optional[hk.initializers.Initializer]=hk.initializers.VarianceScaling(1.0,"fan_avg","uniform"),
            resize_method:str = 'linear',
            dropout_rate:Optional[float] = None,
            fft_branch:bool = True,
            conv_kwargs:dict = {},
            fft_kwargs:dict = {},
            mlp_kwargs:dict = {},
            name:Optional[str] = None,
            **kwargs
    ) -> None:
        """Initialise with
        ---------------------
        - img_shapes: a nested tuple of integers. (b0_shape, b2_shape_layer1,...,b2_shape_layer_last, output_shape). See below for more explainations.
        - b1_channels: a sequency of integers. The number of convolution channels in branch 1, one integer per layer.
        - b2_channels: a sequency of integers. The number of convolution channels in branch 2, one integer per layer.
        - b3_channels: a sequency of integers. The number of convolution channels in branch 3, one integer per layer.
        - activation: function. The activation function for all except the last layers.
        w_init: haiku initalizer. Weight initializer.
        - resize_method: string. Resize method availbel in jax.image.resize. default linear.
        - dropout_rate: a float between 0 and 1. default None.
        - fft_branch: bool. When True, Fourier transform is applied. If false, there will be no Fourier layers. 
        - conv_kwargs: for haiku convolution layers.
        - fft_kwargs: for jax.numpy.fft.rfftn and irffn.
        - mlp_kwargs: for flowrec.model.feedforward.MLP

        Parameter 'img_shapes' and channels
        ------------------------
        This is a nested tuple of integers, each tuple specifies the shape of the image the network will resize to at layers (b0, b2_1, b2_2, ... , b2_last, output_shape). The shape of images in branch 1 will be the same as the shape 'b2_last'. 
        - For example, if we want the output to be a 128-by-128 flow field with 3 variables, and we want three layers in u-net style branch 2 but only one layer is branch 3 after merging.
        ima_shapes = ((64,64), (32,32), (16,16), (32,32), (128,128))
        b3_channels = (3,)
        After branch 0, the image will have shape (64,64), but the Fourier convolution will operate of an image of shape (32,32).


        Available keywords
        -------------------------
        - b0_filter: tuple of integers. Filter size of branch 0 convolution layer. Default (3,3).
        """
        super().__init__(name)

        if 'b0_filter' in kwargs:
            b0_filter = kwargs['b0_filter']
        else:
            b0_filter = (3,3)

        try:
            assert isinstance(img_shapes[0], tuple)
            assert len(img_shapes) > 2
        except AssertionError as e:
            logger.error("Wrong user value for 'img_shapes'.")
            raise e
        fb1 = _test_cnn_filters(b1_filters, b1_channels)
        fb2 = _test_cnn_filters(b2_filters, b2_channels)
        fb3 = _test_cnn_filters(b3_filters, b3_channels)
        self.act = activation
        self.dropout_rate = dropout_rate
        self.b0_shape = img_shapes[0]
        assert isinstance(self.b0_shape[0],int)
        self.b2_shapes = img_shapes[1:-1]
        assert isinstance(self.b2_shapes[0],tuple)
        self.b1_shape = self.b2_shapes[-1]
        assert isinstance(self.b1_shape[0],int)
        self.output_shape = img_shapes[-1]
        assert isinstance(self.output_shape[0],int)
        if len(self.b2_shapes) != len(fb2):
            print(len(self.b2_shapes),len(fb2))
            raise ValueError("The expected number of layers in branch 2 is different when calculating from 'image_shapes' and 'b2_channels'.")
        self.output_dim = b3_channels[-1]
        logger.debug(f'Output will have {self.output_dim} variables.')
        
        # define functions for later
        v_resize = jax.vmap(Partial(jax.image.resize,method=resize_method),(-1,None),-1)
        self.vv_resize = jax.vmap(v_resize,(0,None),0)

        ## DEFINE NETWORK
        # define mlp network
        mlp_size = np.prod(np.asarray(self.b0_shape+(self.output_dim,))).astype('int16')
        self._mlp = MLP([mlp_size], activation=self.act, w_init=w_init, dropout_rate=self.dropout_rate, **mlp_kwargs)

        # branch 0: first convolution
        self._b0conv = hk.Conv2D(self.output_dim, b0_filter, w_init=w_init, name='branch0_conv',**conv_kwargs)

        # branch 1: image does not change size, fourier if requested
        b1 = []
        if fft_branch:
            fft_axes = list(range(0, len(self.output_shape)+1))
            self._fft = Partial(jnp.fft.rfftn, axes=fft_axes, **fft_kwargs)
            self._ifft = Partial(jnp.fft.irfftn, axes=fft_axes, **fft_kwargs)
            logger.debug(f'Branch 1 is the Fourier branch. Creating FFT and inverse FFT function. Taking fourier transform over axes {fft_axes}')
        else:
            self._fft = _empty_fun
            self._ifft = _empty_fun
            logger.debug('No Fourier branch.')

        for i, (c,f) in enumerate(zip(b1_channels, fb1)):
            b1.append(
                hk.Conv2D(c, f, name=f'branch1_conv_{i}', **conv_kwargs)
            )
        logger.info("'w_init' is not specified in branch 1.")
        self.b1 = tuple(b1)

        # branch 2: unet style
        b2 = []
        for i, (c,f) in enumerate(zip(b2_channels, fb2)):
            b2.append(
                hk.Conv2D(c, f, w_init=w_init, name=f'branch2_conv_{i}', **conv_kwargs)
            )
        self.b2 = tuple(b2)

        # merge branches
        b3 = []
        for i, (c,f) in enumerate(zip(b3_channels, fb3)):
            b3.append(
                hk.Conv2D(c, f, w_init=w_init, name=f'branch3_conv_{i}', **conv_kwargs)
            )
        self.b3 = tuple(b3)
        logger.info(f'Branch 1 has {len(self.b1)} layers, branch 2 has {len(self.b2)} layers, Branch 3 (after merging has) {len(self.b3)} layers.')


    def __call__(self, x, TRAINING):
        '''Apply network to x. No dropout if TRAINING is false. Activation function is not applied after the last layer.'''
        if TRAINING:
            logger.info('Model is called in training mode.')
        else:
            logger.info('Model is called in prediction mode.')

        # branch 0
        x = self._mlp(x,TRAINING).reshape((-1,)+self.b0_shape+(self.output_dim,))
        logger.debug(f'First reshape to {x.shape}')
        x = self.act(x)
        x = self._b0conv(x)
        x = self.act(x)
        x = self.dropout(x, TRAINING, self.dropout_rate)
        logger.debug(f'The inital branch (branch 0) produced an image shape {x.shape}')

        ## split branch here
        # branch 1 fourier
        x1 = self.vv_resize(x, self.b1_shape)
        x1 = self._fft(x1)
        logger.debug(f'Output shape of the FFT is {x1.shape}')
        for layer in self.b1:
            x1 = layer(x1)
            x1 = self.dropout(x1, TRAINING, self.dropout_rate)
            x1 = self.act(x1)
        x1 = self._ifft(x1)
        logger.debug(f'Branch 1 produced an image shape {x1.shape}')

        # branch 2 cnn
        x2 = self.vv_resize(x, self.b2_shapes[0])
        logger.debug(f'Branch 2 first resize to shape {x2.shape}')
        x2 = self.b2[0](x2)
        x2 = self.dropout(x2,TRAINING, self.dropout_rate)
        x2 = self.act(x2)
        for new_shape, conv in zip(self.b2_shapes[1:], self.b2[1:]):
            x2 = self.vv_resize(x2, new_shape)
            x2 = conv(x2)
            x2 = self.dropout(x2,TRAINING, self.dropout_rate)
            x2 = self.act(x2)
        
        ## Merge branch (branch 3)
        x3 = jnp.concatenate((x1,x2), axis=-1)
        x3 = self.vv_resize(x3, self.output_shape)
        logger.debug(f'Merging branch 1 and 2, and resizing to the output shape {self.output_shape}')
        for b3layer in self.b3[:-1]:
            x3 = b3layer(x3)
            x3 = self.dropout(x3,TRAINING,self.dropout_rate)
            x3 = self.act(x3)
        x3 = self.b3[-1](x3)

        return x3
        
    @staticmethod
    def dropout(x, TRAINING:bool, dropout_rate:Optional[float]):
        """Apply dropout with if TRAINING is True"""
        if TRAINING and (dropout_rate is not None):
            logger.debug('Doing dropout')
            return hk.dropout(hk.next_rng_key(), dropout_rate, x)
        else:
            return x




class Model(BaseModel):
    """Container for the Fourier2Branch model.
    
    A two branch model, branch 1 is fourier convolution, branch 2 is a normal convolution.

    """
    def __init__(
        self,
        img_shapes:NestedTupleInteger,
        b1_channels:Sequence[int],
        b2_channels:Sequence[int],
        b3_channels:Sequence[int],
        b1_filters:List[Tuple[int, ...]] = [(3,3),],
        b2_filters:List[Tuple[int, ...]] = [(3,3),],
        b3_filters:List[Tuple[int, ...]] = [(3,3),],
        dropout_rate:Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        def forward_fn(x, TRAINING=True):
            mdl = Fourier2Branch(
                img_shapes=img_shapes,
                b1_channels=b1_channels,
                b2_channels=b2_channels,
                b3_channels=b3_channels,
                b1_filters=b1_filters,
                b2_filters=b2_filters,
                b3_filters=b3_filters,
                dropout_rate=dropout_rate,
                **kwargs
            )
            return mdl(x, TRAINING)
        
        self.mdl = hk.transform(forward_fn)
        self._apply = jax.jit(self.mdl.apply,static_argnames='TRAINING')
        self._init = jax.jit(self.mdl.init)
        self._predict = jax.jit(jax.tree_util.Partial(self.mdl.apply,TRAINING=False))
        logger.info('Successfully created model.')

        
    def init(self, rng, sample_input) -> hk.Params:
        '''Initialise params'''
        params = self._init(rng,sample_input)
        return params
    
    def apply(self, params:hk.Params, rng:jax.random.PRNGKey, *args, **kwargs):
        '''hk.Transformed.apply, training mode by default\n
        
        Arguments:\n
            params: hk.Params.\n
            rng: jax random number generator key.\n
            Also takes positional and keyword arguments for hk.Transformed.apply.
        '''
        return self._apply(params, rng, *args, **kwargs)
    
    def predict(self,params:hk.Params,x,**kwargs):
        '''Same as apply, but Training flag is False and no randomness.'''
        return self._predict(params,None,x,**kwargs)