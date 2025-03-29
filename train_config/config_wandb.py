from ml_collections import config_dict
from ml_collections.config_dict import placeholder
import warnings


_valid_loss_fn = ['physicswithdata_mae', 'physicsnoreplace_mae', 'physicsreplacemean_mae', 'physicsandmean_mae', 'physicswithdata', 'physicsnoreplace', 'physicsreplacemean', 'physicsandmean']
_valid_loss_fn2 = ['mse',]

def get_config(cfgstr:str = None):
    
    if cfgstr:
        user = dict([x.split('@') for x in cfgstr.split(',')])
    else:
        user = {}
        # warnings.warn('No training case is selected, proceeds with the basic configuration. Are you sure this is not a mistake?')

    # Set up default options
    _mdl = 'slice3d'
    _loss_fn = 'physicswithdata'
    if 'model' in user:
        _mdl = user['model']
    if 'loss_fn' in user:
        _loss_fn = user['loss_fn']

    cfg = config_dict.ConfigDict()

    cfg.mode = 'disabled' # online, offline, or disabled. Disabled means no runs will be created.
    cfg.project = 'FlowReconstruction'
    cfg.entity = placeholder(str)
    cfg.group = placeholder(str)
    cfg.job_type = placeholder(str)
    cfg.name = placeholder(str)
    cfg.tags = placeholder(tuple)
    cfg.save_code = placeholder(bool)
    cfg.reinit = placeholder(bool)
    cfg.notes = placeholder(str)
    cfg.resume = placeholder(str)
    cfg.id = placeholder(str)
    cfg.config_exclude_keys = placeholder(tuple)

    ## config to pass to wandbinit.config
    cfg.config = config_dict.ConfigDict()

    cfg.config.log_frequency = 10 # must be an integer

    # training
    cfg.config.batch_size = placeholder(int)
    cfg.config.dropout_rate = placeholder(float)
    cfg.config.regularisation_strength = placeholder(float)
    cfg.config.learning_rate = placeholder(float)

    # cases
    cfg.config._case_dataloader = placeholder(str)
    cfg.config._case_observe = placeholder(str)
    cfg.config._case_select_model = placeholder(str)
    cfg.config._case_loss_fn = placeholder(str)

    # data
    cfg.config.shuffle = placeholder(bool)
    cfg.config.shuffle_batch = placeholder(bool)
    cfg.config.remove_mean = placeholder(bool)
    cfg.config.normalise = placeholder(bool)
    cfg.config.re = placeholder(float)
    cfg.config.snr = placeholder(float)
    cfg.config.filter = placeholder(str)
    cfg.config.percent_observed = placeholder(float)

    # model general
    cfg.config.activation = placeholder(str)

    # not wandb init items
    cfg.use_artifact = placeholder(str)

    if _mdl == 'ffcnn':
        cfg.config.mlp_layers = placeholder(tuple)
        cfg.config.cnn_channels = placeholder(tuple)
        cfg.config.cnn_filters = placeholder(tuple)
    elif _mdl == 'fc2branch':
        cfg.config.img_shapes = placeholder(tuple)
        cfg.config.b1_channels = placeholder(tuple)
        cfg.config.b2_channels = placeholder(tuple)
        cfg.config.b3_channels = placeholder(tuple)
        cfg.config.b1_filters = placeholder(tuple)
        cfg.config.b2_filters = placeholder(tuple)
        cfg.config.b3_filters = placeholder(tuple)
        cfg.config.fft_branch = placeholder(bool)
    elif _mdl == 'ff':
        cfg.config.mlp_layers = placeholder(tuple)
    elif _mdl == 'slice3d':
        cfg.config.load_state = placeholder(str)
        cfg.config.newvar_model = placeholder(str)
        cfg.config.newvar_config = placeholder(str)
        cfg.config.reduce_layers = placeholder(tuple)
    else:
        raise ValueError('Invalid model option for wandb configuration.')

    if _loss_fn in _valid_loss_fn:
        cfg.config.weight_momentum = placeholder(float)
        cfg.config.weight_continuity = placeholder(float)
        cfg.config.weight_sensors = placeholder(float)
    elif _loss_fn in _valid_loss_fn2:
        pass
    else:
        raise ValueError('Invalid loss option for wandb configuration.')


    return cfg

