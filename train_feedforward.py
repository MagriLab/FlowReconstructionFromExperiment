import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from pathlib import Path
import numpy as np
import h5py
import jax.numpy as jnp
import jax
import haiku as hk
import optax

from models.data import data_partition
from utils import simulation2d as project
from utils import training_state as state_utils
from utils.training_state import TrainingState

import time

train_test_split = [600,100,100]
learning_rate = 0.0001
epochs = 160000
# epochs = 1

print("Started at: ", time.asctime(time.localtime(time.time())))

# ======================= pre-processing =========================
(ux,uy,pp) = project.read_data(Path("./local_data/re100"),132)

[ux_train,ux_val,ux_test], [ux_train_m,ux_val_m,ux_test_m] = data_partition(ux,0,train_test_split,REMOVE_MEAN=True)
[uy_train,uy_val,uy_test], [uy_train_m,uy_val_m,uy_test_m] = data_partition(uy,0,train_test_split,REMOVE_MEAN=True)
pb = pp[:,0,49:80] # pressure at the base
[pb_train,pb_val,pb_test], [pb_train_m,pb_val_m,pb_test_m] = data_partition(pb,0,train_test_split,REMOVE_MEAN=True)

(nt,nx,ny) = ux.shape

# ==================== define network ============================

# set up model
def feedforward(x):
    mlp = hk.nets.MLP([200,500,nx*ny],
                        activation=jax.nn.tanh,
                        w_init=hk.initializers.VarianceScaling(1.0,"fan_avg","uniform"))
    return mlp(x)
mdl = hk.transform(feedforward)
mdl = hk.without_apply_rng(mdl)


# set up optimiser
optimizer = optax.adamw(learning_rate, weight_decay=1e-5)

# define loss
@jax.jit
def loss(params,x,y):
    pred = mdl.apply(params,x)
    l = jnp.mean((pred - y)**2)
    return l

@jax.jit
def update(state: TrainingState, x, y):
    l, grads = jax.value_and_grad(loss)(state.params, x, y)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return l, TrainingState(params, opt_state)


# ======================== train ==============================

def fit(x_train,y_train,x_val,y_val,state,epochs):
    loss_train = []
    loss_val = []

    best_state = state
    min_loss = np.inf
    for i in range(1,epochs):
        
        xx_train = jax.random.permutation(jax.random.PRNGKey(i),
                                            x_train,axis=0,
                                            independent=False)
        yy_train = jax.random.permutation(jax.random.PRNGKey(i),
                                            y_train,axis=0,
                                            independent=False)

        l, state = update(state, xx_train, yy_train)
        loss_train.append(l)

        l_val = loss(state.params,x_val,y_val)
        loss_val.append(l_val)        
        if l_val < min_loss:
            best_state = state
            min_loss = l_val

        if i%200 == 0:
            print(f'epoch: {i}, loss: {l:.7f}')
    state = best_state

    return state, loss_train, loss_val


# training ux
rng = jax.random.PRNGKey(1)
params = mdl.init(rng,pb_train[0,:]) # initalise weights
opt_state = optimizer.init(params)
ux_state = TrainingState(params, opt_state)

ux_state, ux_loss_train, ux_loss_val = fit(pb_train, 
                                            ux_train,
                                            pb_val,
                                            ux_val,
                                            ux_state,
                                            epochs)

state_utils.save(Path("./local_results"),ux_state,"ux_state")


# training uy
rng = jax.random.PRNGKey(2)
params = mdl.init(rng,pb_train[0,:]) # initalise weights
opt_state = optimizer.init(params)
uy_state = TrainingState(params, opt_state)

uy_state, uy_loss_train, uy_loss_val = fit(pb_train, 
                                            uy_train,
                                            pb_val,
                                            uy_val,
                                            uy_state,
                                            epochs)

state_utils.save(Path("./local_results"),uy_state,"uy_state")

with h5py.File(Path("./local_results/results.h5"),'w') as hf:
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

    hf.create_dataset("ux_train_m", data=ux_train_m)
    hf.create_dataset("ux_val_m", data=ux_val_m)
    hf.create_dataset("ux_test_m", data=ux_test_m)

    hf.create_dataset("uy_train_m", data=uy_train_m)
    hf.create_dataset("uy_val_m", data=uy_val_m)
    hf.create_dataset("uy_test_m", data=uy_test_m)

    hf.create_dataset("pb_train_m", data=pb_train_m)
    hf.create_dataset("pb_val_m", data=pb_val_m)
    hf.create_dataset("pb_test_m", data=pb_test_m)

print("Finished at: ", time.asctime(time.localtime(time.time())))