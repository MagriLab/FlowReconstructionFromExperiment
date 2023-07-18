import optax


def get_optimizer(traincfg):
    
    optimisers = []

    scheduler = get_scheduler(traincfg.lr_scheduler,traincfg.learning_rate)

    adam = optax.adamw(
        learning_rate=scheduler,
        weight_decay=traincfg.regularisation_strength
    )
    optimisers.append(adam)
    
    if traincfg.gradient_clip:
        optimisers.append(optax.clip(traincfg.gradient_clip))

    optimizer = optax.chain(*optimisers)
    return optimizer


def get_scheduler(scheduler,lr):

    if scheduler == 'constant':
        return optax.constant_schedule(lr)
    if scheduler == 'exponential_decay':
        return optax.exponential_decay(init_value=lr,transition_steps=1000,decay_rate=0.95,transition_begin=5000,end_value=0.5*lr)
