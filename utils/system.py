from jax.lib import xla_bridge
import sys

def on_which_platform():
    print(xla_bridge.get_backend().platform)


# https://github.com/abseil/abseil-py/issues/99
# Work around TensorFlow's absl.logging depencency which alters the
# default Python logging output behavior when present.
def temporary_fix_absl_logging(level:str = 'warning'):
    if 'absl.logging' in sys.modules:
        import absl.logging
        absl.logging.set_verbosity(level)
        absl.logging.set_stderrthreshold(level)
        # and any other apis you want, if you want