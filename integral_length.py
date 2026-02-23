import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./flowrec/utils/a4.mplstyle')

from scipy.signal import correlate2d

from flowrec.utils import my_discrete_cmap
from flowrec.data import DataMetadata

with h5py.File('./local_data/kolmogorov/dim2_re34_k32_f4_dt1_grid128_14635.h5') as hf:
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

state_u = state[...,:-1]
ufluc = state_u - np.mean(state_u, axis=0, keepdims=True)
grid_shape = (128,128)

c = np.zeros(ufluc.shape)
nt = ufluc.shape[0]
o1, o2 = np.ceil(grid_shape[0]/2).astype('int')-1, np.ceil(grid_shape[1]/2).astype('int')-1
for it in range(0,nt):
    if it % 10 == 0:
        print(it)
    _cuu = correlate2d(ufluc[it,:,:,0], ufluc[it,:,:,0], mode='same', boundary='wrap')
    # _cuu = _cuu / _cuu[o1,o2]
    _cvv = correlate2d(ufluc[it,:,:,1], ufluc[it,:,:,1], mode='same', boundary='wrap')
    # _cvv = _cvv / _cvv[o1,o2]
    c[it,:,:,0] = _cuu
    c[it,:,:,1] = _cvv
c_n = np.sum(c, axis=0)
c_n = c_n / c_n[o1,o2,:]

x1, x2 = np.indices(grid_shape)
dx1 = (x1 - o1) * datainfo_ref.dx
dx2 = (x2 - o2) * datainfo_ref.dy
r = (dx1**2 + dx2**2)**0.5

unique_values, unique_count =  np.unique(r, return_counts=True)

c_r = np.zeros((len(unique_values),2))
for i in range(len(unique_values)):
    idx = r == unique_values[i]
    if np.count_nonzero(idx) != unique_count[i]:
        print(np.count_nonzero(idx), unique_count[i])
        break
    c_r[i,:] = np.einsum('ru -> u', c_n[idx,:]) / unique_count[i] # c_r is u(x, t)u(x+r, t) averaged over x and t. Has shape [r,u].


plt.figure(figsize=(4,2.5))
plt.plot(unique_values, c_r[:,0], label='$\\rho_{11}$', color=my_discrete_cmap(0), linewidth=1, alpha=0.8)
plt.plot(unique_values, c_r[:,1], label='$\\rho_{22}$', color=my_discrete_cmap(1), linewidth=1, alpha=0.8)
plt.hlines(0.0, 0, unique_values[-1], colors='k', linewidth=1, linestyles='dashed')
plt.legend()
plt.xlim([0,unique_values[-1]])
plt.xlabel('$r$')
plt.ylabel('$\\rho(r)$')
plt.savefig('2dkol-twopoint-correlation.pdf', bbox_inches='tight')



first_zero = [False,False]
for i in range(c_r.shape[0]):
    if c_r[i,0] <= 0.0 and not first_zero[0]:
        first_zero[0] = i
    if c_r[i,1] <= 0.0 and not first_zero[1]:
        first_zero[1] = i
l_int_x1 = np.trapz(c_r[:first_zero[0],0], unique_values[:first_zero[0]])
l_int_x2 = np.trapz(c_r[:first_zero[1],1], unique_values[:first_zero[1]])

with open('2dkol-integral-length-scale.txt','x') as f:
    f.write(f"The integral length scale l_u and l_v are {l_int_x1/np.pi:.3f}\pi and {l_int_x2/np.pi:.3f}\pi.")

