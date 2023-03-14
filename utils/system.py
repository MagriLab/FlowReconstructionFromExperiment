from jax.lib import xla_bridge

def on_which_platform():
    print(xla_bridge.get_backend().platform)
