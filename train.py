

WANDB = None
OFFLINE_WANDB = None
DEBUG = None
NUM_GPU = None
MEM = None


def fit(train_config):
    pass

def save_config(config):
    pass

def save_results(config):
    pass

def wandb_init(wandb_config):
    pass






def main(_):
    data, datainfo = config.case.dataloader(data_config)
    take_observation, insert_observation = config.case.observe(data_config)
    data = model.prep_data(data,config)

    mdl = model.make_model(config)
    optimiser = config.optimiser(optimiser_config)

    loss_fn = make_loss_fn(**config)
    loss_fn_validate = partial(loss_fn)
    update = generate_update_fn(loss_fn)

    if WANDB:
        wandb_init(config)

    fit(train_config)

    if WANDB:
        pass

    save_results()
    save_config()