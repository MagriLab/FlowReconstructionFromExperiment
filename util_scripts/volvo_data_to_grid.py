import time
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate, spatial
from pathlib import Path
matplotlib.use('agg')
plt.style.use('./flowrec/utils/ppt.mplstyle')



def interprbf(x, n, points, data):
    x1,x2,y1,y2,z1,z2 = x
    x = np.linspace(x1,x2,n[0])
    y = np.linspace(y1,y2,n[1])
    z = np.linspace(z1,z2,n[2])
    mesh = np.meshgrid(x,y,z,indexing='ij')
    nt,nx,nu = data.shape
    tmp2 = []
    mesh_flatten = np.transpose(np.array(mesh).reshape((3,-1))) # (3,nx)
    for t in tqdm(range(nt)):
        new = interpolate.RBFInterpolator(points[:,:], data[t,:,:], kernel='thin_plate_spline')(mesh_flatten)
        # thin plate spline is smooth up to second derivatives
        tmp2.append(
            new.reshape((len(x),len(y),len(z),nu))
        )
    tmp2 = np.array(tmp2)
    return tmp2, x, y, z

def interptree(x, n, points, data):
    x1,x2,y1,y2,z1,z2 = x
    x = np.linspace(x1,x2,n[0])
    y = np.linspace(y1,y2,n[1])
    z = np.linspace(z1,z2,n[2])
    mesh = np.meshgrid(x,y,z,indexing='ij')
    nt,nx,nu = data.shape
    tmp1 = np.einsum('txu -> tux', data).reshape((nu*nt,nx))
    tmp2 = []
    mesh_flatten = np.transpose(np.array(mesh).reshape((3,-1))) # (nx,3)
    tree = spatial.cKDTree(data=points)
    distance, idx = tree.query(mesh_flatten,15) 
    inverse_distance = 1./distance # (nx n)
    w = inverse_distance / np.einsum('nm -> n',inverse_distance)[:,None] # [nx,n] n is the number of neighbours
    for i in tqdm(range(nt*nu)):
        neighbours = tmp1[i,idx] 
        weighted = np.einsum('xn,xn -> x', neighbours, w)
        tmp2.append(
            weighted
        )
    tmp2 = np.array(tmp2).reshape((nt,nu,len(x),len(y),len(z)))
    return np.einsum('tu... -> t...u', tmp2), x, y, z


def main(args):
    rawdatapath = Path(args.raw_datapath)
    newdatapath = Path(args.new_datapath)
    assert rawdatapath.exists(), 'Raw data path not found.'
    assert not newdatapath.exists(), 'New data path already exist.'
    method = args.method
    x1,x2,y1,y2,z1,z2 = args.x
    draft = args.draft
    
    ## load raw data
    data = np.load(Path(rawdatapath,'data.npy'))
    points = np.load(Path(rawdatapath,'points.npy'))
    t = np.load(Path(rawdatapath,'time.npy'))
    rho = np.load(Path(rawdatapath,'density.npy'))

    # plot cropped area
    fig, (ax0,ax1) = plt.subplots(1,2,figsize=(7,3))
    im0 = ax0.scatter(points[:,0], points[:,2], c=data[0,:,0], s=1, marker='s')
    plt.colorbar(im0)
    ax0.set(xlabel='x', ylabel='y')
    ax0.hlines([z1,z2],x1,x2,'k')
    ax0.vlines([x1,x2],z1,z2,'k')
    im1 = ax1.scatter(points[:,0], points[:,1], c=data[0,:,0], s=1, marker='s')
    ax1.set(xlabel='x',ylabel='y')
    plt.colorbar(im1)
    ax1.hlines([y1,y2],x1,x2,'k')
    ax1.vlines([x1,x2],y1,y2,'k')
    if draft:
        figpath = Path(newdatapath.parent,'draft_bounding_box.png')
        fig.savefig(figpath)
        x = np.linspace(x1,x2,args.n[0])
        y = np.linspace(y1,y2,args.n[1])
        z = np.linspace(z1,z2,args.n[2])
        print(f"Expected dx:{x[1]-x[0]} m, dy:{y[1]-y[0]} m, dz:{z[1]-z[0]} m.")
        return
    else:
        newdatapath.mkdir()
        fig.savefig(Path(newdatapath,'bounding_box.png'))


    # filter data in range
    mask = (points[:, 0] >= x1) & (points[:, 0] <= x2) & \
       (points[:, 1] >= y1) & (points[:, 1] <= y2) & \
       (points[:, 2] >= z1) & (points[:, 2] <= z2)
    newpoints = points[mask]
    newdata = data[:,mask,:]
    newdensity = rho[:,mask]
    density = np.mean(newdensity)

    # interpolate
    _start_time = time.time()
    match method:
        case 'rbf':
            interpdata, x, y, z = interprbf(args.x, args.n, newpoints, newdata)
        case 'tree':
            interpdata, x, y, z = interptree(args.x, args.n, newpoints, newdata)
    _end_time = time.time()
    exec_time = _end_time-_start_time
    print(f"Interpolation took {exec_time // 3600}h {(exec_time % 3600) // 60}min {exec_time % 60 :.1f}s.")
    
    # save data
    np.savez(Path(newdatapath,'data'), x=x, y=y, z=z, t=t, data=interpdata, density=density)
    np.save(Path(newdatapath,'data'), interpdata)
    with open(Path(newdatapath,'info.txt'),'x') as f:
        f.write(f'Vertices of the bounding box (x1,x2,y1,y2,z1,z2) are {args.x}.\n')
        f.write(f"{newdata.shape[1]} points found within the bounding box.\n")
        f.write(f"Maximum density: {newdensity.max():.4f}, minimum density: {newdensity.min()}.\n")
        f.write(f"The data is interpolated onto a {'-by-'.join([str(i) for i in args.n])} regular grid using {method}.\n")
        f.write(f"Grid size dx:{x[1]-x[0]} m, dy:{y[1]-y[0]} m, dz:{z[1]-z[0]} m.")
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolate a cropped region of the volvo rig data onto a regular grid.')
    parser.add_argument('raw_datapath', type=str)
    parser.add_argument('new_datapath', type=str)
    parser.add_argument('-x', type=float, nargs=6, help='Bounding box coordinates [x1,x2,y1,y2,z1,z2]', required=True)
    parser.add_argument('-n', type=int, nargs=3, help='Number of grid points in x, y and z directions', required=True)
    parser.add_argument('--method', type=str, help='Interpolation method.', required=False, default='rbf')
    parser.add_argument('--draft', action=argparse.BooleanOptionalAction, help="If 'True', save the figures without saving the data.", required=False, default=False)
    args = parser.parse_args()
    main(args)
