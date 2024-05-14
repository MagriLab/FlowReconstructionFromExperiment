from ast import literal_eval


def sweep_preprocess_cfg(cfg):
    # example: cfg._layer1=100, cfg._layer2=200
    # should become cfg.layers=[100,200] 

    # t = literal_eval(cfg._cnn_filters)
    # cfg.update({'cnn_filters':t}, allow_val_change=True)
    # t2 = literal_eval(cfg._cnn_channels)
    # cfg.update({'cnn_channels':t2}, allow_val_change=True)

    # b1_channels = literal_eval(cfg._b1_channels)
    # b1_filters = literal_eval(cfg._b1_filters)
    b2_filters = literal_eval(cfg._b2_filters)

    cfg.update(
        {
        'b2_filters': b2_filters,
        },
        allow_val_change=True
    )
    
    return cfg
