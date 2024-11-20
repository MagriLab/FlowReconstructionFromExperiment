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
            'dropout_rate': 0.005,
            'fft_branch': False,
            'b1_channels': (4,4)
            }
        datacfg_update = {'normalise': False}
        traincfg_update = {
            'learning_rate': 0.0023,
            'nb_batches': 9,
            'regularisation_strength': 0.05,
            'weight_continuity': 2.1,
            'weight_sensors': 11.0,
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



def clean_minimum(group, sensor_randseed=None, loss_fn=None):

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
        randseed_in = int(sensor_randseed)
        randseed_sensor = 10*int(sensor_randseed)+83
        cfgstr = f'dataloader@2dkol,observe@random_pin,loss_fn@{loss_fn}'
        datacfg_update = {
            'random_sensors': (randseed_sensor, 150),
            'random_input': (randseed_in, 80),
        }
        if loss_fn == 'physicsnoreplace':
            mdlcfg_update = {
                'b1_channels': (4,4),
                'b1_filters': ((5,5),),
                'b2_filters': ((5,5),),
                'fft_branch': False,
            }
            traincfg_update = {
                'lr_scheduler': 'cyclic_decay_default',
                'learning_rate': 0.004,
                'nb_batches': 33,
                'weight_sensors': 8.0
            }
        elif loss_fn == 'physicsreplacemean':
            mdlcfg_update = {
                'b1_channels': (4,),
                'b1_filters': ((5,5),),
                'b2_filters': ((5,5),),
                'fft_branch': False,
            }
            traincfg_update = {
                'lr_scheduler': 'cyclic_decay_default',
                'learning_rate': 0.0047,
                'nb_batches': 10,
                'weight_sensors': 64.0
            }
        elif loss_fn == 'physicswithdata':
            mdlcfg_update = {
                'b1_channels': (1,),
                'b1_filters': ((5,5),),
                'b2_filters': ((5,5),),
                'fft_branch': True,
            }
            traincfg_update = {
                'lr_scheduler': 'cyclic_decay_default',
                'learning_rate': 0.0008,
                'nb_batches': 38,
            }
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update
    

def noise_2dkol(testcase:str, snr:int, sensor_randseed:int):
    randseed_in = sensor_randseed
    randseed_sensor = 10*sensor_randseed + 83
    datacfg_update = {
        'random_sensors': (randseed_sensor, 150),
        'random_input': (randseed_in, 80),
    }
    cfgstr = 'loss_fn@' + testcase
    match testcase:
        case 'physicsnoreplace': # physicsnoreplace, classic
            if snr == 20:
                # artifact 'yaxinm/FlowReconstruction/sweep_weights_d4ygq05m:v3'
                # 'yaxinm/FlowReconstruction/run-nl5vcioh-history:v0'
                mdlcfg_update = {
                    'b1_channels': (4,4),
                    'b1_filters': ((5,5),),
                    'b2_filters': ((5,5),),
                    'fft_branch': False,
                }
                traincfg_update = {
                    'lr_scheduler': 'cyclic_decay_default',
                    'learning_rate': 0.004,
                    'nb_batches': 33,
                    'weight_sensors': 8.0
                }
                datacfg_update.update({
                    'normalise': True
                })
            elif snr == 10:
                # artifact 'yaxinm/FlowReconstruction/sweep_weights_uulkhg5w:v2'
                mdlcfg_update = {
                    'b1_channels': (4,4),
                    'b1_filters': ((5,5),),
                    'fft_branch': False,
                    'dropout_rate': 0.0024,
                }
                traincfg_update = {
                    'lr_scheduler': 'cyclic_decay_default',
                    'learning_rate': 0.00034,
                    'nb_batches': 28,
                    'weight_sensors': 26.1,
                    'regularisation_strength': 0.0012,
                }
            elif snr == 5:
                # artifact 'yaxinm/FlowReconstruction/sweep_weights_hisb655c:v8'
                mdlcfg_update = {
                    'b1_channels': (4,4),
                    'b1_filters': ((5,5),),
                    'fft_branch': False,
                    'dropout_rate': 0.0061,
                }
                traincfg_update = {
                    'lr_scheduler': 'cyclic_decay_default',
                    'learning_rate': 0.00018,
                    'nb_batches': 24,
                    'weight_sensors': 7.6,
                    'regularisation_strength': 0.0037,
                }
            else:
                raise NotImplementedError
        case 'physicswithdata': # physicswithdata, loss3
            if snr == 20:
                mdlcfg_update = {
                    'b1_channels': (1,),
                    'b1_filters': ((5,5),),
                    'b2_filters': ((5,5),),
                    'fft_branch': True,
                }
                traincfg_update = {
                    'lr_scheduler': 'cyclic_decay_default',
                    'learning_rate': 0.0008,
                    'nb_batches': 38,
                }
            elif snr ==10:
                # artifact 'sweep_weights_2n0251mx:v1'
                mdlcfg_update = {
                    'b1_channels': (1,),
                    'b1_filters': ((5,5),),
                    'b2_filters': ((5,5),),
                    'fft_branch': True,
                    'dropout_rate': 0.00585
                }
                traincfg_update = {
                    'lr_scheduler': 'cyclic_decay_default',
                    'learning_rate': 0.0028,
                    'nb_batches': 28,
                    'regularisation_strength': 0.008
                }
                datacfg_update.update({
                    'normalise': True
                })
            elif snr == 5:
                # artifact 'yaxinm/FlowReconstruction/sweep_weights_j9tcf74x:v1'
                mdlcfg_update = {
                    'b1_channels': (1,),
                    'b1_filters': ((5,5),),
                    'b2_filters': ((3,3),),
                    'fft_branch': True,
                    'dropout_rate': 0.00167
                }
                traincfg_update = {
                    'lr_scheduler': 'cyclic_decay_default',
                    'learning_rate': 0.00877,
                    'nb_batches': 28,
                    'regularisation_strength': 0.0056
                }
            else:
                raise NotImplementedError
        case 'physicsreplacemean': # physicsreplacemean, lossmean
            if snr == 20:
                # artifacts: 'yaxinm/FlowReconstruction/sweep_weights_7u2kjpr9:v5'
                # 'yaxinm/FlowReconstruction/run-fhw70v4z-history:v0' 
                mdlcfg_update = {
                    'b1_channels': (4,),
                    'b1_filters': ((5,5),),
                    'b2_filters': ((5,5),),
                    'fft_branch': False,
                }
                traincfg_update = {
                    'lr_scheduler': 'cyclic_decay_default',
                    'learning_rate': 0.0047,
                    'nb_batches': 10,
                    'weight_sensors': 64.0
                }
                datacfg_update.update({
                    'normalise': False
                })
            elif snr ==10:
                # artifact 'sweep_weights_h0mh8urd:v1'
                mdlcfg_update = {
                    'b1_channels': (4,),
                    'b1_filters': ((5,5),),
                    'b2_filters': ((3,3),),
                    'fft_branch': False,
                    'dropout_rate': 0.0021
                }
                traincfg_update = {
                    'lr_scheduler': 'cyclic_decay_default',
                    'learning_rate': 0.0013,
                    'nb_batches': 7,
                    'weight_sensors': 34.65,
                    'regularisation_strength': 0.0007
                }
                datacfg_update.update({
                    'normalise': False
                })
            elif snr == 5:
                # artifact 'sweep_weights_grce0t1e:v7'
                mdlcfg_update = {
                    'b1_channels': (4,),
                    'b1_filters': ((5,5),),
                    'b2_filters': ((3,3),),
                    'fft_branch': False,
                    'dropout_rate': 0.0023
                }
                traincfg_update = {
                    'lr_scheduler': 'cyclic_decay_default',
                    'learning_rate': 0.0032,
                    'nb_batches': 5,
                    'weight_sensors': 5.9,
                    'regularisation_strength': 0.0039
                }
                datacfg_update.update({
                    'normalise': False
                })
            else:
                raise NotImplementedError
        case _:
            raise NotImplementedError

    return cfgstr, mdlcfg_update, datacfg_update, traincfg_update


def extreme_events(num_sensors:int, snr:int, sensor_randseed:int):
    datacfg_update = {
        'random_sensors': (sensor_randseed, num_sensors),
        'pressure_inlet_slice': ((8,None,16),(8,None,16)),
        'data_dir': "./local_data/kolmogorov/dim2_re40_k32_dt1_T800_grid128_586178_short.h5",
        'train_test_split': (6000,200,200),
        'randseed': 93658,
        'snr': snr,
        're': 40,
    }

    match snr:
        case 10:
            mdlcfg_update = {
                'b1_channels': (1,),
                'dropout_rate': 0.001,
                'fft_branch': True,
                'b2_filters': ((5,5),),
            }
            traincfg_update ={
                'nb_batches': 30,
                'learning_rate': 0.001,
                'regularisation_strength': 0.0005,
                'lr_scheduler': 'cyclic_decay_default',
            }
        case _:
            raise ValueError
    
    return '', mdlcfg_update, datacfg_update, traincfg_update

def get_config(cfgstr:str):

    experiment = dict([x.split('@') for x in cfgstr.split(',')])
    # objective@noise,group@10,case@1

    objective_config_str = {
        'noise-2dtriangle': 'dataloader@2dtriangle,model@fc2branch,observe@random_pin,',
        'clean_minimum': 'model@fc2branch,',
        'noise-2dkol': 'dataloader@2dkol,model@fc2branch,observe@random_pin,',
        'extreme-events': 'dataloader@2dkol,model@fc2branch,observe@random_pin',
    }

    objective = experiment['objective']
    general_cfgstr = objective_config_str[objective]

    match objective:
        case 'noise-2dtriangle':
            print("running experiment 'noise-2dtriangle'")

            testcase = {
                '1': lossclassic,
                '2': loss3,
                '3': lossmean3
            }

            _cfgstr, mdlcfg_update, datacfg_update, traincfg_update = testcase[experiment['case']]('snr'+experiment['group'])
            
            datacfg_update.update({
                'randseed': 19070949,
                'snr': float(experiment['group']),
                'random_sensors':(136412,250),
            })

        case 'noise-2dkol':
            print("running experiment 'noise-2dkol'")

            testcase = {
                '1': 'physicsnoreplace',
                '2': 'physicswithdata',
                '3': 'physicsreplacemean'
            }

            _cfgstr, mdlcfg_update, datacfg_update, traincfg_update = noise_2dkol(
                testcase=testcase[experiment['case']],
                snr=int(experiment['group']),
                sensor_randseed=int(experiment['sensor_randseed'])
            )

            datacfg_update.update({
                'randseed': 157759,
                'snr': float(experiment['group']),
            })

        case 'clean_minimum':
            print("running experiment 'clean_repeat'")

            testgroup = {
                '1': '2dtriangle',
                '2': '2dkol'
            }

            testcase = {
                '1': 'physicsnoreplace',
                '2': 'physicswithdata',
                '3': 'physicsreplacemean'
            }
            
            _fn_input = {'group':testgroup[experiment['group']]}
            
            if experiment['group'] == '2':
                if 'case' not in experiment or 'sensor_randseed' not in experiment:
                    raise ValueError('Please provide the case number and sensor randseed for the test group you selected.')
                _fn_input.update({'loss_fn': testcase[experiment['case']]})
                _fn_input.update({'sensor_randseed': experiment['sensor_randseed']})
            else:
                if 'case' in experiment:
                    raise NotImplementedError

            _cfgstr, mdlcfg_update, datacfg_update, traincfg_update = clean_minimum(**_fn_input)

        case 'extreme-events':
            print("running experiment 'extreme-events'")

            testcase = {} # case: number of sensors
            testgroup = {} # signal to noise ratio
            _cfgstr, mdlcfg_update, datacfg_update, traincfg_update = extreme_events(
                num_sensors=int(experiment['case']), 
                snr=int(experiment['group']), 
                sensor_randseed=int(experiment['sensor_randseed'])
            )

        case _:
            raise NotImplementedError


    ## get general config
    general_cfgstr = general_cfgstr + _cfgstr
    cfg = base_config.get_config(general_cfgstr)
    
    ## update configs
    cfg.model_config.update(mdlcfg_update)
    cfg.data_config.update(datacfg_update)
    cfg.train_config.update(traincfg_update)

    FLAGS._experimentcfgstr = general_cfgstr

    return cfg
