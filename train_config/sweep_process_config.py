from ast import literal_eval


def sweep_preprocess_cfg(cfg):
    # example: cfg._layer1=100, cfg._layer2=200
    # should become cfg.layers=[100,200] 
    t = literal_eval(cfg._cnn_filters)
    cfg.update({'cnn_filters':t}, allow_val_change=True)
    t2 = literal_eval(cfg._cnn_channels)
    cfg.update({'cnn_channels':t2}, allow_val_change=True)
    
    return cfg
