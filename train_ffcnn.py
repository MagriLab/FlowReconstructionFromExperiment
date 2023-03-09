import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".25"

import sys
import time
import h5py
import logging
logger = logging.getLogger(f'fr.{__name__}')
logger.propagate = False
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter('%(name)s.%(funcName)s:%(lineno)d [%(levelname)s] %(message)s'))
logger.addHandler(_handler)
# logger.setLevel(logging.DEBUG)

# logging.getLogger('fr.flowrec.models.cnn').setLevel(logging.DEBUG)

import numpy as np
from pathlib import Path
from utils import simulation2d as project
from flowrec.data import data_partition, DataMetadata
import flowrec.training_and_states as train
from flowrec.training_and_states import TrainingState
from flowrec import losses
import flowrec.physics_and_derivatives as derivatives
from flowrec.models.cnn import Model
import jax 
import jax.numpy as jnp
import optax
import wandb

WANDB = True
train_test_split = [600,100,100]
learning_rate = 0.00005
mlp_layers = [] # size of the intermediate layers
cnn_channels = [32,16,3]
cnn_filter = [(3,3)]
dropout_rate = 0.00
regularisation_strength = 0.0000
epochs = 80000
nb_batches = 6
re = 100
data_dir = Path("./local_data/re"+str(re))
dt = 0.0002*625
dx = 12/512
dy = 4/128

sensor_slicing_space = np.s_[::15,::5] # taking points as sensors
weighting = [0.1,0.9] # weighting terms for [physics,sensors]
e = 0.01 # when to consider the solution 'converged'

print("Started at: ", time.asctime(time.localtime(time.time())))
time_stamp = time.strftime("%y%m%d%H%M%S",time.localtime(time.time()))
results_dir = f'ffcnn_physics/{time_stamp}'


wandb_group = 'FF_CNN'
wandb_run = f'pi{time_stamp}relu'

# ======================= pre-processing =========================
x_base = 132
(ux,uy,pp) = project.read_data(data_dir,x_base)
x = np.stack([ux,uy,pp],axis=0)
######### delete last rows
x = x[:,:,:250,:]

data_randseed = np.random.randint(1,1000000)
[x_train,x_val,x_test], _ = data_partition(x,1,train_test_split,REMOVE_MEAN=False,randseed=data_randseed,SHUFFLE=False) # Do not shuffle, do not remove mean for training with physics informed loss


[ux_train,uy_train,pp_train] = np.squeeze(np.split(x_train,3,axis=0))
[ux_val,uy_val,pp_val] = np.squeeze(np.split(x_val,3,axis=0))
[ux_test,uy_test,pp_test] = np.squeeze(np.split(x_test,3,axis=0))

triangle_base_coords = [49,80]
pb_train = project.take_measurement_base(pp_train,ly=triangle_base_coords,centrex=0)
pb_val = project.take_measurement_base(pp_val,ly=triangle_base_coords,centrex=0)
pb_test = project.take_measurement_base(pp_test,ly=triangle_base_coords,centrex=0)

(nt,nx,ny) = ux_train.shape
n_base = pb_train.shape[-1]

# information about the grid
datainfo = DataMetadata(
    re = re,
    discretisation=[dt,dx,dy],
    axis_index=[0,1,2],
    problem_2d=True
)

mlp_layers.extend([3*nx*ny])

pb_train = np.reshape(pb_train,(train_test_split[0],-1))
pb_val = np.reshape(pb_val,(train_test_split[1],-1))


u_train = np.stack((ux_train,uy_train,pp_train),axis=-1)
u_val = np.stack((ux_val,uy_val,pp_val),axis=-1)

# take sensor measurement
sensor_slicing = (np.s_[:],) + sensor_slicing_space

observed_train = u_train[sensor_slicing]
observed_val = u_val[sensor_slicing]

percent_observed = 100*(observed_train[0,...].size/u_train[0,...].size)


# =================== model ===============
rng = jax.random.PRNGKey(np.random.randint(10,50))
optimizer = optax.adamw(learning_rate=learning_rate,weight_decay=regularisation_strength)

mdl = Model(mlp_layers,output_shape=(nx,ny,3),cnn_channels=cnn_channels,cnn_filters=cnn_filter,dropout_rate=dropout_rate)


# ==================== Define loss function ====================
# loss_fn = losses.loss_mse
def loss_fn(apply_fn,params,rng,x,y,w=[0.5,0.5],e=0.0001,**kwargs):
    pred = apply_fn(params, rng, x, **kwargs)
    logger.debug(f'Prediction has shape {pred.shape}')
    loss_div = losses.divergence(pred[...,0], pred[...,1], datainfo)
    loss_mom_x = jnp.mean(derivatives.momentum_residue_field(
                            1,
                            ux=pred[...,0],
                            uy=pred[...,1],
                            p=pred[...,2],
                            datainfo=datainfo)**2
                            )
    loss_mom_y = jnp.mean(derivatives.momentum_residue_field(
                            2,
                            ux=pred[...,0],
                            uy=pred[...,1],
                            p=pred[...,2],
                            datainfo=datainfo)**2
                            )
    loss_sensor = losses.mse(pred[sensor_slicing], y)

    return w[0]*jax.nn.relu(loss_div+loss_mom_x+loss_mom_y-e) + w[1]*loss_sensor, (loss_div,loss_mom_x+loss_mom_y,loss_sensor)
    # return w[0]*(loss_div+(loss_mom_x+loss_mom_y-e)**2) + w[1]*loss_sensor, (loss_div,loss_mom_x+loss_mom_y,loss_sensor)


mdl_validation_loss = jax.jit(jax.tree_util.Partial(loss_fn,mdl.apply,TRAINING=False,w=weighting,e=e))

# update = train.generate_update_fn(mdl.apply,optimizer,loss_fn) # this update weights once.
# update = train.generate_update_fn(mdl.apply,optimizer,loss_fn) # this update weights once.
update = train.generate_update_fn(mdl.apply,optimizer,loss_fn,kwargs_value_and_grad={'has_aux':True},kwargs_loss={'w':weighting,'e':e}) # this update weights once.


# ===================== weights & biases ======================
if WANDB:
    wandb_config = {
        "learning_rate": learning_rate,
        "layers": mlp_layers,
        "number_of_layers": len(mlp_layers)+len(cnn_channels),
        "cnn_filter": cnn_filter,
        "activation": "tanh",
        "loss_fn": "mse",
        "dropout_rate":dropout_rate,
        "l2_strength":regularisation_strength,
        "Re": re,
        "weight_physics":weighting[0],
        "weight_sensors":weighting[1],
        "percent_observed":percent_observed,
        "convergence_threshold":e
    }
    run = wandb.init(config=wandb_config,
                project="FlowReconstruction",
                group=wandb_group,
                name=wandb_run,
                tags=['GridSensors']
    )
    run.save('train_ffcnn.py')
    if weighting[0] > 0.0:
        run.tags = run.tags + ('PhysicsInformed',)


# ====================== training =========================
# def fit(x_train,
#         y_train,
#         x_val,
#         y_val,
#         state,
#         epochs,
#         rng,
#         batch=1):
#     loss_train = []
#     loss_val = []

#     best_state = state
#     min_loss = np.inf
#     for i in range(1,epochs+1):

#         [rng] = jax.random.split(rng,1)
        
#         # xx_train = jax.random.permutation(rng,
#         #                                     x_train,
#         #                                     axis=0,
#         #                                     independent=False)
#         # yy_train = jax.random.permutation(rng,
#         #                                     y_train,
#         #                                     axis=0,
#         #                                     independent=False) # slow
                        
#         # xx_batched = jnp.array_split(xx_train,batch,axis=0)
#         # yy_batched = jnp.array_split(yy_train,batch,axis=0) # slow

#         xx_batched = jnp.array_split(x_train,batch,axis=0)
#         yy_batched = jnp.array_split(y_train,batch,axis=0) # slow

#         loss_epoch = []
#         for b in range(batch):
#             l, state = update(state, rng, xx_batched[b], yy_batched[b])
#             if dropout_rate is None:
#                 loss_epoch.append(l)
#             else:
#                 l = mdl_validation_loss(state.params,None,xx_batched[b],yy_batched[b])
#                 loss_epoch.append(l)
#             logger.debug(f'batch size is {xx_batched[b].shape[0]}')
#             logger.info(f'batch: {b}, loss: {l:.7f}')
#         loss_train.append(np.mean(loss_epoch))
        

#         l_val = mdl_validation_loss(state.params,None,x_val,y_val)
#         loss_val.append(l_val)        

#         if WANDB:
#             wandb.log({'loss':l, 'loss_val':l_val})
        
#         if l_val < min_loss:
#             best_state = state
#             min_loss = l_val
#             train.save_trainingstate(Path('./local_results',results_dir),state,'state')

#         if i%200 == 0:
#             print(f'epoch: {i}, loss: {loss_train[i-1]:.7f}, validation_loss: {l_val:.7f}', flush=True)
#     state = best_state

#     return state, loss_train, loss_val




def fit(
    x_train_batched,
    y_train_batched,
    x_val,
    y_val,
    state,
    epochs,
    rng,
    n_batch,
):

    loss_train = []
    loss_val = []
    loss_div = []
    loss_momentum = []
    loss_sensors = []

    best_state = state
    min_loss = np.inf

    for i in range(epochs+1):
        [rng] = jax.random.split(rng,1)

        loss_epoch = []
        loss_epoch_div = []
        loss_epoch_mom = []
        loss_epoch_s = []
        for b in range(n_batch):
            (l, (l_div, l_mom, l_s)), state = update(state, rng, x_train_batched[b], y_train_batched[b])
            if dropout_rate is None:
                loss_epoch.append(l)
                loss_epoch_div.append(l_div)
                loss_epoch_mom.append(l_mom)
                loss_epoch_s.append(l_s)
            else:
                l, (l_div, l_mom, l_s) = mdl_validation_loss(state.params,None,x_train_batched[b],y_train_batched[b])
                loss_epoch.append(l)
                loss_epoch_div.append(l_div)
                loss_epoch_mom.append(l_mom)
                loss_epoch_s.append(l_s)
            logger.debug(f'batch size is {x_train_batched[b].shape[0]}')
            logger.debug(f'batch: {b}, loss: {l:.7f}.')
        loss_train.append(np.mean(loss_epoch))
        loss_div.append(np.mean(loss_epoch_div))
        loss_momentum.append(np.mean(loss_epoch_mom))
        loss_sensors.append(np.mean(loss_epoch_s))

        
        l_val, _ = mdl_validation_loss(state.params,None,x_val,y_val)
        loss_val.append(l_val)


        if WANDB:
            wandb.log({'loss':loss_train[-1], 'loss_val':l_val, 'loss_div':loss_div[-1], 'loss_momentum':loss_momentum[-1], 'loss_sensors':loss_sensors[-1]})

        if l_val < min_loss:
            best_state = state
            min_loss = l_val
            train.save_trainingstate(Path('./local_results',results_dir),state,'state')

        if i%200 == 0:
            print(f'Epoch: {i}, loss: {loss_train[-1]:.7f}, validation_loss: {l_val:.7f}', flush=True)
            print(f'For training, loss_div: {loss_div[-1]:.7f}, loss_momentum: {loss_momentum[-1]:.7f}, loss_sensors: {loss_sensors[-1]:.7f}')
    state = best_state

    return state, loss_train, loss_val, (loss_div, loss_momentum, loss_sensors)


# start training
params = mdl.init(rng,pb_train[0,:]) # initalise weights
logger.debug(jax.tree_util.tree_map(lambda x: x.shape,params))
opt_state = optimizer.init(params)
state = TrainingState(params, opt_state)
# state, loss_train, loss_val = fit(
                                # pb_train,
                                # u_train,
                                # pb_val,
                                # u_val,
                                # state,
                                # epochs,
                                # rng,
                                # batch=nb_batches)


xx_batched = jnp.array_split(pb_train,nb_batches,axis=0)
yy_batched = jnp.array_split(observed_train,nb_batches,axis=0)
state, loss_train, loss_val, (loss_div, loss_momentum, loss_sensors) = fit(
    xx_batched,
    yy_batched,
    pb_val,
    observed_val,
    state,
    epochs,
    rng,
    nb_batches
)

if WANDB:
    run.finish()
    
# ======================= save results ========================

with h5py.File(Path(f'./local_results/{results_dir}/results.h5'),'w') as hf:
    hf.create_dataset("loss_train",data=np.array(loss_train))
    hf.create_dataset("loss_val",data=np.array(loss_val))
    hf.create_dataset("loss_div",data=np.array(loss_div))
    hf.create_dataset("loss_momentum",data=np.array(loss_momentum))
    hf.create_dataset("loss_sensors",data=np.array(loss_sensors))

    hf.create_dataset("data_randseed",data=data_randseed)
    hf.create_dataset('train_test_split',data=train_test_split)
    hf.create_dataset('x_base',data=x_base)
    hf.create_dataset('triangle_base_coords',data=triangle_base_coords)
    hf.create_dataset("data_dir",data=data_dir.absolute().as_posix())

with h5py.File(Path(f'./local_results/{results_dir}/parameters.h5'),'w') as hf:
    hf.create_dataset("mlp_layers",data=mlp_layers)
    hf.create_dataset("cnn_channels",data=list(cnn_channels))
    hf.create_dataset("cnn_filter",data=np.array(cnn_filter))
    hf.create_dataset("learning_rate",data=learning_rate)

print("Finished at: ", time.asctime(time.localtime(time.time())))
