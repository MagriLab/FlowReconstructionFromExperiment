from flowrec._typing import *
from ml_collections.config_dict import ConfigDict
from utils import simulation2d
from utils.py_helper import slice_from_tuple
from flowrec.data import data_partition

import jax


def dataloader_2dtriangle(cfg:ConfigDict) -> dict:
    # u_train (tensor), u_val (tensor), inn_train (vector), inn_val (vector)
    # datainfo
    x_base = 132
    triangle_base_coords = [49,80]

    (ux,uy,pp) = simulation2d.read_data(cfg.data_dir,x_base)
    x = np.stack([ux,uy,pp],axis=0)
    # remove parts where uz is not zero
    s = slice_from_tuple(cfg.slice_to_keep)
    x = x[s]

    if cfg.SHUFFLE:
        randseed = np.random.randint(1,10000)
    else:
        randseed = None

    
    [x_train,x_val,_], _ = data_partition(x,1,cfg.train_test_split,REMOVE_MEAN=cfg.REMOVE_MEAN,randseed=randseed,SHUFFLE=cfg.SHUFFLE) # Do not shuffle, do not remove mean for training with physics informed loss

    [ux_train,uy_train,pp_train] = np.squeeze(np.split(x_train,3,axis=0))
    [ux_val,uy_val,pp_val] = np.squeeze(np.split(x_val,3,axis=0))
    
    pb_train = simulation2d.take_measurement_base(pp_train,ly=triangle_base_coords,centrex=0)
    pb_val = simulation2d.take_measurement_base(pp_val,ly=triangle_base_coords,centrex=0)

    (nt,nx,ny) = ux_train.shape

    # information about the grid
    datainfo = DataMetadata(
        re = cfg.re,
        discretisation=[cfg.dt,cfg.dx,cfg.dy],
        axis_index=[0,1,2],
        problem_2d=True
    ).to_named_tuple()

    pb_train = np.reshape(pb_train,(cfg.train_test_split[0],-1))
    pb_val = np.reshape(pb_val,(cfg.train_test_split[1],-1))

    u_train = np.stack((ux_train,uy_train,pp_train),axis=-1)
    u_val = np.stack((ux_val,uy_val,pp_val),axis=-1)

    cfg.update({'randseed':randseed})

    data = {
        'u_train': u_train, # [t,x,y,3]
        'u_val': u_val, # [t,x,y,3]
        'inn_train': pb_train, # [t,len]
        'inn_val': pb_val # [t,len]
    }

    return data, datainfo




def observe_grid(data_config:ConfigDict):
    s = slice_from_tuple(data_config.slice_to_keep)

    def take_observation(*args) -> jax.Array:
        out = []
        for u in args:
            out.append(u[s])

        return (*out,)


    def insert_observation(pred:jax.Array, observed:jax.Array) -> jax.Array:
        return pred.at[s].set(observed)
    
    return take_observation, insert_observation



def select_model_ffcnn(mdl_config,data_config):

    def prep_data(data,config) -> dict:
        # make data into suitable form
        pass

    def make_model(model_config) -> BaseModel:
        pass






def make_loss_fn(**kwargs):
    measure  = kwargs['take_observation']
    insert_measurement = kwargs['insert_observation']
    w = kwargs['w']
    def loss_fn(apply_fn,params,rng,x,y,**kwargs):
        pred = apply_fn(params, rng, x, **kwargs)
        pred_measure = measure(pred)
        loss_sensor = some_loss(pred_measure, y)

        pred_new = insert_measurement(pred,y)
        
        # calculate mode loss
        return w[0]*loss_sensor 
    
    return loss_fn