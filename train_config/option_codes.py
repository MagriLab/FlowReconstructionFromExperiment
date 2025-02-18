import sys
import logging
logger = logging.getLogger(f'fr.{__name__}')
logger.propagate = False
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter('%(name)s.%(funcName)s:%(lineno)d [%(levelname)s] %(message)s'))
logger.addHandler(_handler)

from ml_collections.config_dict import ConfigDict


code_dataloader = {
    '2dtriangle': 't2',
    '2dkol': 'k2',
    '3dvolvo': 'tv',
    '3dkol': 'k3',
    '3dkolsets': 'km',
}
code_observe = {
    'grid': 'gd',
    'grid_pin': 'gp',
    'sparse': 'sa',
    'sparse_pin': 'sp',
    'random_pin': 'rp',
    'slice': 'pl',
    'slice_pin': 'pp',
}
code_model = {
    'ffcnn': 'fc',
    'fc2branch': 'b2',
    'ff': 'ff',
}
code_loss = {
    'physicswithdata': 'pi3',
    'physicsnoreplace': 'pi1',
    'physicsreplacemean': 'pm3',
    'physicsandmean': 'pm1',
    'physicswithdata_mae': 'pi3(1)',
    'physicsnoreplace_mae': 'pi1(1)',
    'physicsreplacemean_mae': 'pm3(1)',
    'physicsandmean_mae': 'pm1(1)',
    'mse': 'mse',
}


def code(cfgcase: ConfigDict):
    try:
        sdata = code_dataloader[cfgcase._case_dataloader]
    except KeyError as e:
        sdata = 'zz'
        logger.warning(f"This dataloader '{e}' has not been assigned an option code yet.")

    try: 
        sob = code_observe[cfgcase._case_observe]
    except KeyError as e:
        sob = 'zz'
        logger.warning(f"This observation function '{e}' has not been assigned an option code yet.")
    
    try:
        smdl = code_model[cfgcase._case_select_model]
    except KeyError as e:
        smdl = 'zz'
        logger.warning(f"This model '{e}' has not been assigned an option code yet.")

    try:
        sloss = code_loss[cfgcase._case_loss_fn]
    except KeyError as e:
        sloss = 'zz'
        logger.warning(f"This loss function '{e}' has not been assigned an option code yet.")

    s = sdata + sob + smdl + sloss

    return s

