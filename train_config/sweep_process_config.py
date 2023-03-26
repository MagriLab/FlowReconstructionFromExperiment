from ast import literal_eval


def sweep_preprocess_cfg(cfg):
    # example: cfg._layer1=100, cfg._layer2=200
    # should become cfg.layers=[100,200] 
    if cfg._nb_cnn_layers==1:
        cfg.update({'cnn_channels':(3,)}, allow_val_change=True)
    elif cfg._nb_cnn_layers==2:
        cfg.update({'cnn_channels':(16,3)}, allow_val_change=True)
    elif cfg._nb_cnn_layers==3:
        cfg.update({'cnn_channels':(32,16,3)}, allow_val_change=True)
    
    t = literal_eval(cfg._cnn_filters)
    cfg.update({'cnn_filters':t}, allow_val_change=True)
    
    return cfg