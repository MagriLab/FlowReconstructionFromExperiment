import logging
import sys

_logger = logging.getLogger(f'fr.{__name__}')
_logger.propagate = False
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter('%(name)s.%(funcName)s:%(lineno)d [%(levelname)s] %(message)s'))
if not _logger.handlers:
    _logger.addHandler(_handler)