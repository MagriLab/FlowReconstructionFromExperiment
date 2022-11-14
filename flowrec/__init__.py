import logging
import sys

_logger = logging.getLogger(f'fr.{__name__}')
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter('%(name)s.%(funcName)s:%(lineno)d [%(levelname)s] %(message)s'))
_logger.addHandler(_handler)

def get_models_logger():
    '''Returns the logger used for the module "models".'''
    return _logger