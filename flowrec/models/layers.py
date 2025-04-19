import haiku as hk
import flax.linen as nn
import jax.numpy as jnp
import haiku.experimental.flax as hkflax

from typing import Optional, Callable, Sequence, List


class ElementWiseMultiplication(hk.Module):
    """Perform elementwise multiplication W*X + b"""
    def __init__(self, shape:tuple, with_bias:bool = True, name:str|None = None):
        super().__init__(name)
        self.shape = shape
        self.with_bias = with_bias
    def __call__(self, x):
        w = hk.get_parameter("w", self.shape, init=jnp.ones)
        w = jnp.broadcast_to(w, x.shape)
        if self.with_bias:
            b = hk.get_parameter("b", self.shape, init=jnp.zeros)
            b = jnp.broadcast_to(b, x.shape)
        return w*x + b

class MyConv():
    """Use ND convolution layer from flax in haiku."""
    def __init__(self, 
        output_channels:int,
        kernel_shape:int|Sequence[int],
        stride:int = 1,
        padding:str = 'SAME',
        w_init:Callable = nn.initializers.lecun_normal(),
        name:str = None,
        **kwargs
    ):

        self.conv = hkflax.lift(
            nn.Conv(
                output_channels,
                kernel_shape,
                stride,
                padding=padding,
                kernel_init=w_init,
                **kwargs
            ),
            name = name
        )
    def __call__(self, x):
        return self.conv(x)