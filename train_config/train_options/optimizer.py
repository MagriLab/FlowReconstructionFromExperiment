import optax


def get_optimizer(traincfg):
    adam = optax.adamw(
        learning_rate=traincfg.learning_rate,
        weight_decay=traincfg.regularisation_strength
    )
    clip = optax.clip(1)

    optimizer = optax.chain(
        clip,
        adam
    )
    return optimizer