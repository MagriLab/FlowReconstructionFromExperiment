from ast import literal_eval


def sweep_preprocess_cfg(cfg):
    # example: cfg._layer1=100, cfg._layer2=200
    # should become cfg.layers=[100,200] 
    # t = literal_eval(cfg._cnn_filters)
    # cfg.update({'cnn_filters':t}, allow_val_change=True)
    # t2 = literal_eval(cfg._cnn_channels)
    # cfg.update({'cnn_channels':t2}, allow_val_change=True)

    b1_channels = literal_eval(cfg._b1_channels)
    b2_channels = literal_eval(cfg._b2_channels)
    _img_shapes = {
        3: ((32,32,32),(16,16,16),(4,4,4),(8,8,8),(64,64,64)),
        4: ((32,32,32),(16,16,16),(4,4,4),(8,8,8),(16,16,16),(64,64,64))
    }
    b3_filters = literal_eval(cfg._b3_filters)

    cfg.update(
        {
        'b1_filters': b1_channels,
        'b2_channels': b2_channels,
        'img_shapes': _img_shapes[len(b2_channels)],
        'b3_filters': b3_filters,
        },
        allow_val_change=True
    )
    
    return cfg
