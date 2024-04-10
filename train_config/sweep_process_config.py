from ast import literal_eval


def sweep_preprocess_cfg(cfg):
    # example: cfg._layer1=100, cfg._layer2=200
    # should become cfg.layers=[100,200] 

    # t = literal_eval(cfg._cnn_filters)
    # cfg.update({'cnn_filters':t}, allow_val_change=True)
    # t2 = literal_eval(cfg._cnn_channels)
    # cfg.update({'cnn_channels':t2}, allow_val_change=True)

    # b1_channels = literal_eval(cfg._b1_channels)
    scheduler = {
        'constant': 'constant',
        'exponential_decay': 'exponential_decay',
        'cyclic_decay': "{'scheduler':'cyclic_cosine_decay_schedule','decay_steps':(800,1000,1200,1500),'alpha':(0.3,0.3,0.38,0.38),'lr_multiplier':(1.0,0.75,0.5,0.5),'boundaries':(1000,2200,3600,5500)}"
    }

    cfg.update(
        {
        # 'b1_channels': b1_channels,
        'lr_scheduler': scheduler[cfg._lr_scheduler]
        },
        allow_val_change=True
    )
    
    return cfg
