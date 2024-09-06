import sys
import os
import logging
from jax.lib import xla_bridge
from pathlib import Path
from typing import Union
logger = logging.getLogger(f'fr.{__name__}')

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


def set_gpu(gpu_id:Union[int,str], memory_fraction:float = 1.0):
    '''Set which gpu to use and the memory fraction using CUDA_VISIBLE_DEVICES'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)


def change_cwd(new_cwd: Union[str, Path]):
    """Change the current working directory.
    
    --------------------------
    To change the current working directory to a file's directory, use change_cwd(__file__)
    """
    abspath = Path(new_cwd).resolve()
    dname = abspath.parent
    os.chdir(dname)
    logger.warning(f'The current working directory has been changed to {dname}.')