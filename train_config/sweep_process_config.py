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
    # _img_shapes = {
    #     3: ((32,32,32),(16,16,16),(4,4,4),(8,8,8),(64,64,64)),
    #     4: ((32,32,32),(16,16,16),(4,4,4),(8,8,8),(16,16,16),(64,64,64))
    # }
    # b3_filters = literal_eval(cfg._b3_filters)

    n_val_batch = 500 // cfg.batch_size
    val_batch_idx = tuple(range(-n_val_batch, 0))

    img_shapes3d = literal_eval(cfg._img_shapes3d)
    channels3d = {
          2: (16,4),
          3: (8,8,4)
    }
    filters3d = {
          2: (3,5),
          3: (3,5,5)
    }
    
    cfg.update(
        {
        'b1_channels': b1_channels,
        'b2_channels': b2_channels,
        'img_shapes3d': img_shapes3d,
        'channels3d' : channels3d[len(img_shapes3d)],
        'filters3d': filters3d[len(img_shapes3d)],
        'val_batch_idx': val_batch_idx
        # 'img_shapes': _img_shapes[len(b2_channels)],
        # 'b3_filters': b3_filters,
        },
        allow_val_change=True
    )
    
    return cfg
