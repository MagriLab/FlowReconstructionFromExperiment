import logging
logger = logging.getLogger(f'fr.{__name__}')
from typing import Tuple, Union
from ml_collections.config_dict import ConfigDict

def slice_from_tuple(tu:Tuple) -> Tuple:
    '''Create a tuple of python slices from a tuple.\n
    
    Argument:\n
        tu: a tuple or several tuples in a tuple.\n
        For example, (1,10,2) will return equivalent to np.s_[1:10:2].
        ((None,None,2),(None,None,3)) will return equivalent to np.s_[::2,::3].\n
    
    Return:\n
        slice that can be used for indexing an array.
    '''
    assert isinstance(tu,tuple)
    if isinstance(tu[0],tuple):
        s = [slice(None)]*len(tu)
        for i in range(len(tu)):
            s[i] = slice(*tu[i])
        s = tuple(s)
    else:
        s = slice(*tu)
    logger.debug(f'Created index {s}.')
    
    return s


def update_matching_keys(
        dict_to_update:Union[dict,ConfigDict], 
        dict_with_values:Union[dict,ConfigDict]) -> dict:
    '''Update a dictionary in place with values from another dictionary where there are matching keys.'''

    for key in dict_with_values.keys():
        if key in dict_to_update:
            logger.debug(f'{key} updated from {dict_to_update[key]} to {dict_with_values[key]}')
            dict_to_update.update({key:dict_with_values[key]}) 