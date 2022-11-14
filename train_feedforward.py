import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".10"
from pathlib import Path
import numpy as np
import h5py
import jax.numpy as jnp
import jax
import optax

from flowrec.data import data_partition
from utils import simulation2d as project
import flowrec.training_and_states as train
from flowrec.training_and_states import TrainingState
from flowrec.models.feedforward import Model as FeedForward
from flowrec import losses
import time
import wandb


train_test_split = [600,100,100]
learning_rate = 0.0001
layers = [31] # size of the intermediate layers
epochs = 8000
# epochs = 5

print("Started at: ", time.asctime(time.localtime(time.time())))
results_dir = time.strftime("%y%m%d%H%M",time.localtime(time.time()))

# ======================= pre-processing =========================
(ux,uy,pp) = project.read_data(Path("./local_data/re100"),132)
x = np.stack([ux,uy,pp],axis=0)

[x_train,x_val,x_test], [xm_train,xm_val,xm_test] = data_partition(x,1,train_test_split,REMOVE_MEAN=True)

[ux_train,uy_train,pp_train] = np.squeeze(np.split(x_train,3,axis=0))
[ux_val,uy_val,pp_val] = np.squeeze(np.split(x_val,3,axis=0))
[ux_test,uy_test,pp_test] = np.squeeze(np.split(x_test,3,axis=0))

pb_train = pp_train[:,0,49:80] # pressure at the base
pb_val = pp_val[:,0,49:80] # pressure at the base
pb_test = pp_test[:,0,49:80] # pressure at the base

(nt,nx,ny) = ux_train.shape
n_base = pb_train.shape[-1]

layers.extend([nx*ny])

# ==================== define network ============================

# set up model
rng = jax.random.PRNGKey(2)
mdl = FeedForward(layers,rng=rng,APPLY_RNG=False)

# set up optimiser
optimizer = optax.adamw(learning_rate, weight_decay=1e-5)

# define loss
loss_fn = losses.loss_mse
mdl_loss = jax.jit(jax.tree_util.Partial(loss_fn,mdl))

# update function: update 
update = train.generate_update_fn(mdl,optimizer,loss_fn)

# ===================== weights & biases ======================

wandb_config = {
    "learning_rate": learning_rate,
    "layers": layers,
    "number_of_layers": len(layers),
    "activation": "tanh",
    "loss_fn": "mse"
}
run = wandb.init(config=wandb_config,
            project="FlowReconstruction",
            group="FF",
            name=f'2layer-{results_dir}'
)


# ======================== train ==============================

def fit(x_train,y_train,x_val,y_val,state,epochs,f_name,rng):
    loss_train = []
    loss_val = []

    best_state = state
    min_loss = np.inf
    for i in range(1,epochs+1):

        [rng] = jax.random.split(rng,1)
        
        xx_train = jax.random.permutation(rng,
                                            x_train,
                                            axis=0,
                                            independent=False)
        yy_train = jax.random.permutation(rng,
                                            y_train,
                                            axis=0,
                                            independent=False)

        l, state = update(state, xx_train, yy_train)
        loss_train.append(l)

        l_val = mdl_loss(state.params,x_val,y_val)
        loss_val.append(l_val)        
        wandb.log({f'loss_{f_name}':l, 
                    f'loss_val_{f_name}':l_val,
                    f'epochs_{f_name}':i}
        )
        if l_val < min_loss:
            best_state = state
            min_loss = l_val
            train.save_trainingstate(Path(f'./local_results/{results_dir}')
                                    ,state,
                                    f'{f_name}_state')

        if i%200 == 0:
            print(f'epoch: {i}, loss: {l:.7f}, validation_loss: {l_val:.7f}', flush=True)
    state = best_state

    return state, loss_train, loss_val

pb_train = jnp.reshape(pb_train,(train_test_split[0],-1))
pb_val = jnp.reshape(pb_val,(train_test_split[1],-1))

# training ux
print("Start training ux ...")
ux_train = jnp.reshape(ux_train,(train_test_split[0],-1))
ux_val = jnp.reshape(ux_val,(train_test_split[1],-1))
rng = jax.random.PRNGKey(np.random.randint(1,20))
params = mdl.init(pb_train[0,:]) # initalise weights
opt_state = optimizer.init(params)
ux_state = TrainingState(params, opt_state)

ux_state, ux_loss_train, ux_loss_val = fit(pb_train, 
                                            ux_train,
                                            pb_val,
                                            ux_val,
                                            ux_state,
                                            epochs,
                                            "ux",
                                            rng)



# training uy
print("Start training uy ...")
uy_train = jnp.reshape(uy_train,(train_test_split[0],-1))
uy_val = jnp.reshape(uy_val,(train_test_split[1],-1))
rng = jax.random.PRNGKey(np.random.randint(1,20))
params = mdl.init(pb_train[0,:]) # initalise weights
opt_state = optimizer.init(params)
uy_state = TrainingState(params, opt_state)

uy_state, uy_loss_train, uy_loss_val = fit(pb_train, 
                                            uy_train,
                                            pb_val,
                                            uy_val,
                                            uy_state,
                                            epochs,
                                            "uy",
                                            rng)

run.finish()
# ================= save results =======================

pb_train = pb_train.reshape((train_test_split[0],n_base))
pb_val = pb_val.reshape((train_test_split[1],n_base))
ux_train = ux_train.reshape((train_test_split[0],nx,ny))
ux_val = ux_val.reshape((train_test_split[1],nx,ny))
uy_train = uy_train.reshape((train_test_split[0],nx,ny))
uy_val = uy_val.reshape((train_test_split[1],nx,ny))


with h5py.File(Path(f'./local_results/{results_dir}/results.h5'),'w') as hf:
    hf.create_dataset("ux_loss_train",data=np.array(ux_loss_train))
    hf.create_dataset("ux_loss_val",data=np.array(ux_loss_val))
    hf.create_dataset("ux_train", data=ux_train)
    hf.create_dataset("ux_val", data=ux_val)
    hf.create_dataset("ux_test", data=ux_test)
    
    hf.create_dataset("uy_loss_train",data=np.array(uy_loss_train))
    hf.create_dataset("uy_loss_val",data=np.array(uy_loss_val))
    hf.create_dataset("uy_train", data=uy_train)
    hf.create_dataset("uy_val", data=uy_val)
    hf.create_dataset("uy_test", data=uy_test)
    
    hf.create_dataset("pb_train", data=pb_train)
    hf.create_dataset("pb_val", data=pb_val)
    hf.create_dataset("pb_test", data=pb_test)

    hf.create_dataset("ux_train_m", data=xm_train[0,...])
    hf.create_dataset("ux_val_m", data=xm_val[0,...])
    hf.create_dataset("ux_test_m", data=xm_test[0,...])

    hf.create_dataset("uy_train_m", data=xm_train[1,...])
    hf.create_dataset("uy_val_m", data=xm_val[1,...])
    hf.create_dataset("uy_test_m", data=xm_test[1,...])

    hf.create_dataset("pb_train_m", data=xm_train[2,...])
    hf.create_dataset("pb_val_m", data=xm_val[2,...])
    hf.create_dataset("pb_test_m", data=xm_test[2,...])

with h5py.File(Path(f'./local_results/{results_dir}/parameters.h5'),'w') as hf:
    hf.create_dataset("layers",data=layers)
    hf.create_dataset("learning_rate",data=learning_rate)
print("Finished at: ", time.asctime(time.localtime(time.time())))
