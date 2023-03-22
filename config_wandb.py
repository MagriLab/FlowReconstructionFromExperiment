from ml_collections import config_dict
from ml_collections.config_dict import placeholder

def get_config(cfgstr:str = 'ffcnn'):

    cfg = config_dict.ConfigDict()

    cfg.mode = 'online'
    cfg.project = 'FlowReconstruction'
    cfg.group = placeholder(str)
    cfg.name = placeholder(str)
    cfg.tags = placeholder(tuple)
    cfg.save_code = placeholder(bool)
    cfg.reinit = placeholder(bool)
    cfg.notes = placeholder(str)
    cfg.config_exclude_keys = placeholder(tuple)

    ## config to pass to wandbinit.config
    cfg.config = config_dict.ConfigDict()

    cfg.config.mdl = cfgstr
    cfg.config.nb_batches = placeholder(int)
    cfg.config.weight_physics = placeholder(float)
    cfg.config.weight_sensors = placeholder(float)
    cfg.config.re = placeholder(float)
    cfg.config.dropout_rate = placeholder(float)
    cfg.config.regularisation_trength = placeholder(float)
    cfg.config.percent_observed = placeholder(float)


    if cfgstr == 'ffcnn':
        cfg.config.mlp_layers = placeholder(tuple)
        cfg.config.cnn_channels = placeholder(tuple)
        cfg.config.cnn_filter = placeholder(tuple)


    return cfg

