import train_config.config as base_config
from absl import flags

FLAGS = flags.FLAGS

def loss3(casestr:str):
    raise ValueError('Update the defaults first!')

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
    raise ValueError('Update the defaults first!')

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
    raise ValueError('Update the defaults first!')

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

    if group == '2dtriangle':

        cfgstr = 'dataloader@2dtriangle,observe@sparse_pin'
        datacfg_update = {
            'sensor_index': ((52, 52, 27, 27, 78, 27, 48, 114, 9, 9, 160, 200, 220, 230, 187, 232, 10, 10 ),(45, 83, 51, 77, 64, 64, 64, 64, 50, 78, 15, 97, 64, 34, 92, 115, 10, 115)),
            'normalise': False,
        }
        mdlcfg_update = {
            'b1_channels': (1,),
        }
        traincfg_update = {
            'nb_batches': 20,
            'learning_rate': 0.0075,
            'lr_scheduler': "{'scheduler':'cyclic_cosine_decay_schedule','decay_steps':(800,1000,1200,1500,2000),'alpha':(0.3,0.3,0.38,0.38,0.38),'lr_multiplier':(1.0,1.0,1.0,0.7,0.5),'boundaries':(1000,2200,3600,5500,8000)}",
        }

    else:
        raise NotImplementedError
    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update
    



def get_config(cfgstr:str):

    experiment = dict([x.split('@') for x in cfgstr.split(',')])
    # objective@noise,group@10,case@1

    objectives = {
        'noise': 'dataloader@2dtriangle,model@fc2branch,observe@grid_pin,',
        'clean_minimum': 'model@fc2branch,loss_fn@physicswithdata,',
    }

    general_cfgstr = objectives[experiment['objective']]

    if experiment['objective'] == 'noise':
        raise ValueError('Update the defaults first!')
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

    if experiment['objective'] == 'clean_minimum':
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
