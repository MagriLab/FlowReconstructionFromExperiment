import sys
sys.path.append('../')
import h5py
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from flowrec.data import DataMetadata
from flowrec.physics_and_derivatives import derivative1, derivative2
from flowrec.losses import divergence, momentum_loss
from flowrec.utils.system import change_cwd

change_cwd(__path__)


with h5py.File('../local_data/kolmogorov/dim2_re34_k32_f4_dt1_grid128_14635.h5') as hf:
    state = np.array(hf.get('state'))
    dt = float(hf.get('dt')[()])
    re = float(hf.get('re')[()])
    nref = float(hf.get('ngrid')[()])
datainfo_ref = DataMetadata(
    re=re,
    discretisation=[dt,np.pi/nref,np.pi/nref],
    axis_index=[0,1,2],
    problem_2d=True
).to_named_tuple()

v_derivative1 = jax.vmap(derivative1,(0,None,None),0)
v_derivative2 = jax.vmap(derivative2,(0,None,None),0)
# function that applies a function to inn, and x,y,z in order
def didj(de_fun,inn): # the first axis of 'inn' is the velocity
    didj_T = de_fun(inn,datainfo_ref.dx,datainfo_ref.axx).reshape((-1,)+inn.shape)
    for j in range (1,inn.shape[0]):
        didj_T = jnp.concatenate(
            (
            didj_T,
            de_fun(inn,datainfo_ref.discretisation[1:][j],datainfo_ref.axis_index[1:][j]).reshape((-1,)+inn.shape)
            ),
            axis=0
        )
    return didj_T # for de_fun = v_derivative1 and inn=u -> [j,i,t,x,y,z]



## float 64
dui_dxj_64 = []
d2ui_dxj2_64 = []
n = 0
while 50*n < state.shape[0]:
    state_u = jnp.moveaxis(state[50*n:50*(n+1),...,:-1], -1, 0)
    u = state_u.astype(jnp.float64)
    dui_dxj_T_n = didj(v_derivative1,u)
    d2ui_dxj2_n = didj(v_derivative2,u)
    dui_dxj_64.append(jnp.einsum('jitxy -> tijxy',dui_dxj_T_n))    
    d2ui_dxj2_64.append(jnp.einsum('jitxy -> tijxy',d2ui_dxj2_n))    
    n += 1
dui_dxj_64 = jnp.concatenate(dui_dxj_64, axis=0)
d2ui_dxj2_64 = jnp.concatenate(d2ui_dxj2_64, axis=0)
print(dui_dxj_64[0,0,0,0,0])



## float 32
dui_dxj_32 = []
d2ui_dxj2_32 = []
n = 0
while 100*n < state.shape[0]:
    state_u = jnp.moveaxis(state[100*n:100*(n+1),...,:-1], -1, 0)
    u = state_u.astype(jnp.float32)
    dui_dxj_T_n = didj(v_derivative1,u)
    d2ui_dxj2_n = didj(v_derivative2,u)
    dui_dxj_32.append(jnp.einsum('jitxy -> tijxy',dui_dxj_T_n))    
    d2ui_dxj2_32.append(jnp.einsum('jitxy -> tijxy',d2ui_dxj2_n))    
    n += 1
dui_dxj_32 = jnp.concatenate(dui_dxj_32, axis=0)
d2ui_dxj2_32 = jnp.concatenate(d2ui_dxj2_32, axis=0)
print(dui_dxj_32[0,0,0,0,0])


diff1 = jnp.abs(dui_dxj_64.astype(jnp.float32) - dui_dxj_32)
diff2 = jnp.abs(d2ui_dxj2_64.astype(jnp.float32) - d2ui_dxj2_32)
idx1 = jnp.argmax(diff1.flatten())
idx2 = jnp.argmax(diff2.flatten())


print(f"Largest difference in 1st derivative: {diff1.flatten()[idx1]}. Value of this element in float32: {dui_dxj_32.flatten()[idx1]}.")
print(f"Largest difference in 2st derivative: {diff2.flatten()[idx2]}. Value of this element in float32: {d2ui_dxj2_32.flatten()[idx2]}.")



## actual losses
ld64 = jnp.array(0.0, dtype=jnp.float64)
lm64 = jnp.array(0.0, dtype=jnp.float64)
n = 0
while 50*n < state.shape[0]:
    u_p = state[50*n:50*(n+1),...].astype(jnp.float64)
    ld = divergence(u_p[...,:-1].astype(jnp.float64), datainfo_ref)
    lm = momentum_loss(u_p, datainfo_ref)
    ld64 += ld
    lm64 += lm
    n += 1
ld64 = ld64 / n
lm64 = lm64 / n
print(ld64, lm64)

## actual losses
ld32 = jnp.array(0.0, dtype=jnp.float32)
lm32 = jnp.array(0.0, dtype=jnp.float32)
n = 0
while 100*n < state.shape[0]:
    u_p = state[100*n:100*(n+1),...].astype(jnp.float32)
    ld = divergence(u_p[...,:-1].astype(jnp.float32), datainfo_ref)
    lm = momentum_loss(u_p, datainfo_ref)
    ld32 += ld
    lm32 += lm
    n += 1
ld32 = ld32 / n
lm32 = lm32 / n
print(ld32, lm32)

print(f"Difference between the reference divergence loss: {jnp.abs(ld64-ld32)}, and momentum loss: {jnp.abs(lm64-lm32)}.")