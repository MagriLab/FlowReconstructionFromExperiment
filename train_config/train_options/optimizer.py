import logging
import optax

from ast import literal_eval
from flowrec import lr_schedule

logger = logging.getLogger(f'fr.{__name__}')

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
    elif scheduler == 'exponential_decay':
        return optax.exponential_decay(init_value=lr,transition_steps=1000,decay_rate=0.95,transition_begin=5000,end_value=0.5*lr)
    elif scheduler == 'cyclic_decay_default':
        my_scheduler = lr_schedule.cyclic_cosine_decay_schedule(
            init_value=lr,
            decay_steps=(800,1000,1200,1500),
            alpha=(0.3,0.3,0.38,0.38),
            lr_multiplier=(1.0,0.75,0.5,0.5),
            boundaries=(1000,2200,3600,5500)
        )
        return my_scheduler    
    else:
        schedule_kwargs = literal_eval(scheduler)
        my_scheduler_name = schedule_kwargs.pop('scheduler')
        logger.debug(f"Using custom learning rate schedule '{my_scheduler_name}'.")
        my_scheduler = getattr(lr_schedule, my_scheduler_name)
        return my_scheduler(init_value=lr, **schedule_kwargs)

