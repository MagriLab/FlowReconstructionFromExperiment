import sys
import logging

def set_logger():
    logger = logging.getLogger('utils')
    logger.propagate = False
    
    def set_handlers():
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('%(module)s.%(funcName)s:%(lineno)d [%(levelname)s] %(message)s'))
        logger.addHandler(h)
    set_handlers()

    return logger 

def log_level(logger,level):
    logger.setLevel(level)

logger = set_logger()