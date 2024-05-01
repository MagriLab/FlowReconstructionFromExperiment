import train_config.config as base_config
from absl import flags

FLAGS = flags.FLAGS

def loss3(casestr:str):

    cfgstr = 'loss_fn@physicswithdata'

    if casestr == 'snr20':
        mdlcfg_update = {
            'dropout_rate': 0.0048,
            'fft_branch': True,
            }
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.014,
            'nb_batches': 16,
            'regularisation_strength': 0.095,
            'lr_scheduler': 'cyclic_decay_default'
        }
    elif casestr == 'snr10':
        mdlcfg_update = {
            'dropout_rate': 0.0055,
            'fft_branch': True,
            'b1_channels': (1,),
            }
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.0068,
            'nb_batches': 13,
            'regularisation_strength': 0.056,
            'lr_scheduler': 'cyclic_decay_default'
        }
    elif casestr == 'snr5':
        mdlcfg_update = {
            'dropout_rate': 0.006,
            'fft_branch': True,
            'b1_channels': (1,),
            }
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.0088,
            'nb_batches': 13,
            'regularisation_strength': 0.07,
            'lr_scheduler': 'cyclic_decay_default'
        }
    else:
        raise NotImplementedError

    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update
    


def lossclassic(casestr:str):

    cfgstr = 'loss_fn@physicsnoreplace'

    if casestr == 'snr20':
        mdlcfg_update = {
            'dropout_rate': 0.0005,
            'fft_branch': False,
            'b1_channels': (4,4),
            }
        datacfg_update = {'normalise': True}
        traincfg_update = {
            'learning_rate': 0.0013,
            'nb_batches': 13,
            'regularisation_strength': 0.007,
            'weight_continuity': 2.0,
            'weight_sensors': 12.0,
            'lr_scheduler': 'cyclic_decay_default'
        }
    elif casestr == 'snr10':
        mdlcfg_update = {
            'dropout_rate': 0.0077,
            'fft_branch': False,
            'b1_channels': (4,4)
            }
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.0023,
            'nb_batches': 9,
            'regularisation_strength': 0.05,
            'weight_continuity': 2.1,
            'weight_sensors': 1.05,
            'lr_scheduler': 'cyclic_decay_default'
        }
    elif casestr == 'snr5':
        mdlcfg_update = {
            'dropout_rate': 0.0047,
            'fft_branch': False,
            'b1_channels': (4,4)
            }
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.0012,
            'nb_batches': 9,
            'regularisation_strength': 0.022,
            'weight_continuity': 3.8,
            'weight_sensors': 11,
            'lr_scheduler': 'cyclic_decay_default'
        }
    else:
        raise NotImplementedError

    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update
    

    

def lossmean3(casestr:str):

    cfgstr = 'loss_fn@physicsreplacemean'

    if casestr == 'snr20':
        raise NotImplementedError
        # Remove the error line after checking the new results
        mdlcfg_update = {
            'dropout_rate': 0.0022,
            'fft_branch': False,
            'b1_channels': (1,)
            }
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.003,
            'nb_batches': 12,
            'regularisation_strength': 0.076,
            'weight_continuity': 2.5,
            'weight_sensors': 31.5,
            'lr_scheduler': 'cyclic_decay_default'
        }
    elif casestr == 'snr10':
        mdlcfg_update = {
            'dropout_rate': 0.0025,
            'fft_branch': False,
            'b1_channels': (4,4)
            }
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.0023,
            'nb_batches': 3,
            'regularisation_strength': 0.09,
            'weight_continuity': 2.7,
            'weight_sensors': 48.5,
            'lr_scheduler': 'cyclic_decay_default'
        }
    elif casestr == 'snr5':
        mdlcfg_update = {
            'dropout_rate': 0.0097,
            'fft_branch': False,
            'b1_channels': (8,)
            }
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.0046,
            'nb_batches': 5,
            'regularisation_strength': 0.042,
            'weight_continuity': 2.6,
            'weight_sensors': 24.0,
            'lr_scheduler': 'cyclic_decay_default'
        }
    else:
        raise NotImplementedError

    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update



def clean_minimum(group):

    if group == '2dtriangle':

        cfgstr = 'dataloader@2dtriangle,observe@sparse_pin,loss_fn@physicsnoreplace'
        datacfg_update = {
            'sensor_index': ((52, 52, 27, 27, 78, 27, 48, 114, 9, 9, 160, 200, 220, 230, 187, 232, 10, 10 ),(45, 83, 51, 77, 64, 64, 64, 64, 50, 78, 15, 97, 64, 34, 92, 115, 10, 115)),
            'normalise': True,
        }
        mdlcfg_update = {
            'fft_branch': False,
            'b1_channels': (8,),
        }
        traincfg_update = {
            'nb_batches': 17,
            'learning_rate': 0.0023,
            'weight_sensors': 40.0,
            'lr_scheduler': 'cyclic_decay_default'
        }
    elif group == '2dkol':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update
    



def get_config(cfgstr:str):

    experiment = dict([x.split('@') for x in cfgstr.split(',')])
    # objective@noise,group@10,case@1

    objectives = {
        'noise-2dtriangle': 'dataloader@2dtriangle,model@fc2branch,observe@grid_pin,',
        'clean_minimum': 'model@fc2branch,',
    }

    general_cfgstr = objectives[experiment['objective']]

    if experiment['objective'] == 'noise-2dtriangle':
        print("running experiment 'noise-2dtriangle'")

        testcase = {
            '1': lossclassic,
            '2': loss3,
            '3': lossmean3
        }

        _cfgstr, mdlcfg_update, datacfg_update, traincfg_update = testcase[experiment['case']]('snr'+experiment['group'])
        
        general_cfgstr = general_cfgstr + _cfgstr
        cfg = base_config.get_config(general_cfgstr)

        datacfg_update.update({
            'slice_grid_sensors': ((1,None,18),(3,None,12)),
            'randseed': 19070949,
            'snr': float(experiment['group'])
        })

    elif experiment['objective'] == 'clean_minimum':
        print("running experiment 'clean_repeat'")

        testgroup = {
            '1': '2dtriangle',
            '2': '2dkol'
        }
        _cfgstr, mdlcfg_update, datacfg_update, traincfg_update = clean_minimum(testgroup[experiment['group']])

        ## get general config
        general_cfgstr = general_cfgstr + _cfgstr
        cfg = base_config.get_config(general_cfgstr)

    else:
        raise NotImplementedError

    ## update configs
    cfg.model_config.update(mdlcfg_update)
    cfg.data_config.update(datacfg_update)
    cfg.train_config.update(traincfg_update)

    FLAGS._experimentcfgstr = general_cfgstr

    return cfg
