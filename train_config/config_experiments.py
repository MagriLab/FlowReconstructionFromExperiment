import train_config.config as base_config
from absl import flags

FLAGS = flags.FLAGS

def loss3(casestr:str):

    cfgstr = 'loss_fn@physicswithdata'

    if casestr == 'snr20':
        mdlcfg_update = {'dropout_rate': 0.0078}
        datacfg_update = {'normalise': True}
        traincfg_update = {
            'learning_rate': 0.0006,
            'nb_batches': 20,
            'regularisation_strength': 0.097
        }
    elif casestr == 'snr10':
        mdlcfg_update = {'dropout_rate': 0.0078}
        datacfg_update = {'normalise': True}
        traincfg_update = {
            'learning_rate': 0.0006,
            'nb_batches': 20,
            'regularisation_strength': 0.1
        }
    elif casestr == 'snr5':
        mdlcfg_update = {'dropout_rate': 0.02}
        datacfg_update = {'normalise': True}
        traincfg_update = {
            'learning_rate': 0.0006,
            'nb_batches': 20,
            'regularisation_strength': 0.2
        }
    else:
        raise NotImplementedError

    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update
    


def lossclassic(casestr:str):

    cfgstr = 'loss_fn@physicsnoreplace'

    if casestr == 'snr20':
        mdlcfg_update = {'dropout_rate': 0.0078}
        datacfg_update = {'normalise': True}
        traincfg_update = {
            'learning_rate': 0.000146,
            'nb_batches': 20,
            'regularisation_strength': 0.02,
            'weight_sensors': 2.0
        }
    elif casestr == 'snr10':
        mdlcfg_update = {'dropout_rate': 0.0078}
        datacfg_update = {'normalise': True}
        traincfg_update = {
            'learning_rate': 0.000146,
            'nb_batches': 20,
            'regularisation_strength': 0.05,
            'weight_sensors': 2.0
        }
    elif casestr == 'snr5':
        mdlcfg_update = {'dropout_rate': 0.006}
        datacfg_update = {'normalise': True}
        traincfg_update = {
            'learning_rate': 0.000146,
            'nb_batches': 20,
            'regularisation_strength': 0.05,
            'weight_sensors': 1.0
        }
    else:
        raise NotImplementedError

    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update
    

    

def lossmean3(casestr:str):

    cfgstr = 'loss_fn@physicsreplacemean'

    if casestr == 'snr20':
        mdlcfg_update = {'dropout_rate': 0.038}
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.002,
            'nb_batches': 8,
            'regularisation_strength': 0.0048,
            'weight_continuity': 1.09,
            'weight_sensors': 15.0,
            'lr_scheduler': 'exponential_decay'
        }

    elif casestr == 'snr10':
        mdlcfg_update = {'dropout_rate': 0.038}
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.002,
            'nb_batches': 8,
            'regularisation_strength': 0.01,
            'weight_continuity': 1.09,
            'weight_sensors': 10.0,
            'lr_scheduler': 'exponential_decay'
        }
    elif casestr == 'snr5':
        mdlcfg_update = {'dropout_rate': 0.03}
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.002,
            'nb_batches': 8,
            'regularisation_strength': 0.015,
            'weight_continuity': 1.09,
            'weight_sensors': 5.0,
            'lr_scheduler': 'exponential_decay'
        }
    else:
        raise NotImplementedError

    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update



def clean_minimum(group):

    if group == 'grid':

        cfgstr = 'observe@grid_pin'
        datacfg_update = {
            'slice_grid_sensors': ((None,None,30),(None,None,30)),
            'normalise': True,
        }

    elif group == 'sparse':

        cfgstr = 'observe@sparse_pin'
        datacfg_update = {
            'sensor_index': ((44, 44, 23, 23, 140, 68, 40, 103, 5, 5, 20, 20, 150, 175, 200, 225),(81, 47, 51, 77, 64, 64, 64, 64, 80, 48, 78, 50, 10, 119, 10, 119)),
            'normalise': True,
        }

    else:
        raise NotImplementedError
    return cfgstr, datacfg_update
    



def get_config(cfgstr:str):

    experiment = dict([x.split('@') for x in cfgstr.split(',')])
    # objective@noise,group@10,case@1

    objectives = {
        'noise': 'dataloader@2dtriangle,model@ffcnn,observe@grid_pin,',
        'clean_minimum': 'dataloader@2dtriangle,model@ffcnn,loss_fn@physicswithdata,',
    }

    general_cfgstr = objectives[experiment['objective']]

    if experiment['objective'] == 'noise':
        print("running experiment 'noise'")

        testcase = {
            '1': lossclassic,
            '2': loss3,
            '3': lossmean3
        }

        _cfgstr, mdlcfg_update, datacfg_update, traincfg_update = testcase[experiment['case']]('snr'+experiment['group'])
        
        general_cfgstr = general_cfgstr + _cfgstr
        cfg = base_config.get_config(general_cfgstr)

        mdlcfg_update.update({
            'cnn_filters': ((5,5),)
        })
        datacfg_update.update({
            'slice_grid_sensors': ((None,None,15),(None,None,10)),
            'randseed': 19070949,
            'snr': float(experiment['group'])
        })

        cfg.model_config.update(mdlcfg_update)
        cfg.data_config.update(datacfg_update)
        cfg.train_config.update(traincfg_update)
    

    if experiment['objective'] == 'clean_minimum':
        print("running experiment 'clean_repeat'")

        testgroup = {
            '1': 'grid',
            '2': 'sparse'
        }
        _cfgstr, datacfg_update = clean_minimum(testgroup[experiment['group']])

        mdlcfg_update = {
            'cnn_filters': ((5,5),)
        }
        traincfg_update = {
            'learning_rate': 0.001,
            'nb_batches': 19,
            'weight_continuity':1.0,
        }

        ## get config and update
        general_cfgstr = general_cfgstr + _cfgstr
        cfg = base_config.get_config(general_cfgstr)

        cfg.model_config.update(mdlcfg_update)
        cfg.data_config.update(datacfg_update)
        cfg.train_config.update(traincfg_update)

    else:
        raise NotImplementedError

    FLAGS._experimentcfgstr = general_cfgstr

    return cfg
