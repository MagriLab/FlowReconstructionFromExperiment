import optax
import jax.numpy as jnp
from typing import Optional, Sequence


def cyclic_cosine_decay_schedule(
        init_value: float,
        lr_multiplier: Optional[Sequence], 
        decay_steps:Optional[Sequence], 
        alpha:Optional[Sequence], 
        boundaries: Optional[Sequence] = None,
        exponents:Optional[Sequence] = None,
    ):
    ''' A cyclic learning rate schedule constructed by combining cosine decays from Optax. \n

    Arguments: \n
        init_value: the initial learning rate.
        lr_multiplier: a sequence of number that is multiplied with the init_lr to define the starting learning rate of each cycle. If the first number in the sequence is 1.0, then the first cycle starts with the initial learning rate defined by init_lr.\n 
        decay_steps: a sequence of number of steps to apply the decay for each cycle.\n
        alpha: a sequence of multipling factor used to define the minimum learning rate.\n
        boundaries: a sequence of steps defining the starting step of each cycle.
    '''
    
    if not exponents:
        exponents = [1.0]*len(decay_steps)
    if not boundaries:
        boundaries = jnp.cumsum(jnp.asarray(decay_steps))
    lr = [init_value*a for a in lr_multiplier]

    schedule_i = []
    for i in range(len(lr)):
        schedule_i.append(
            optax.cosine_decay_schedule(init_value=lr[i], decay_steps=decay_steps[i], alpha=alpha[i], exponent=exponents[i])
        )
    schedule = optax.join_schedules(schedule_i, boundaries)

    return schedule
    