from ml_collections import config_dict
from ml_collections.config_dict import placeholder

def get_config(cfgstr:str = 'ffcnn,physicswithdata'):

    _mdl, _loss_fn = cfgstr.split(',')

    cfg = config_dict.ConfigDict()

    cfg.mode = 'online'
    cfg.project = 'FlowReconstruction'
    cfg.entity = placeholder(str)
    cfg.group = placeholder(str)
    cfg.job_type = placeholder(str)
    cfg.name = placeholder(str)
    cfg.tags = placeholder(tuple)
    cfg.save_code = placeholder(bool)
    cfg.reinit = placeholder(bool)
    cfg.notes = placeholder(str)
    cfg.config_exclude_keys = placeholder(tuple)

    ## config to pass to wandbinit.config
    cfg.config = config_dict.ConfigDict()

    # training
    cfg.config.nb_batches = placeholder(int)
    cfg.config.dropout_rate = placeholder(float)
    cfg.config.regularisation_trength = placeholder(float)
    cfg.config.learning_rate = placeholder(float)

    # cases
    cfg.config._case_dataloader = placeholder(str)
    cfg.config._case_observe = placeholder(str)
    cfg.config._case_select_model = placeholder(str)
    cfg.config._case_loss_fn = placeholder(str)

    # data
    cfg.config.shuffle = placeholder(bool)
    cfg.config.remove_mean = placeholder(bool)
    cfg.config.normalise = placeholder(bool)
    cfg.config.re = placeholder(float)


    if _mdl == 'ffcnn':
        cfg.config.mlp_layers = placeholder(tuple)
        cfg.config.cnn_channels = placeholder(tuple)
        cfg.config.cnn_filter = placeholder(tuple)
    else:
        raise ValueError('Invalid model option for wandb configuration.')

    if _loss_fn == 'physicswithdata' or 'physicsnoreplace':
        cfg.config.weight_physics = placeholder(float)
        cfg.config.weight_sensors = placeholder(float)
    else:
        raise ValueError('Invalid loss option for wandb configuration.')


    return cfg

