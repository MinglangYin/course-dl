"""
Author: Minglang Yin, minglang_yin@brown.edu
    PINNs (physical-informed neural network) for solving lid-driven flow
        u_t + (u*u_x + v*u_y) - nu*(u_xx + u_yy) = -p_x 
        v_t + (u*v_x + v*v_y) + nu*(v_xx + v_yy) = -p_y

    Input: [t, x, y]
    Output: [u, v, p]
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torch.autograd import grad
from utils import *

# torch.manual_seed(123456)
# np.random.seed(123456)

velocity = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(3, 20))
        self.net.add_module('relu_layer_1', nn.ReLU())
        for i in range(2, 6):
            self.net.add_module('linear_layer_%d' %(i), nn.Linear(20, 20))
            self.net.add_module('relu_layer_%d' %(i), nn.ReLU())
        self.net.add_module('linear_layer_10', nn.Linear(20, 3))

        ## define pressure and viscosity
        self.P = 0
        self.nu = 0.1

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        out = self.net(x)
        u, v, p = out[:,0], out[:, 1], out[:, 2]
        
        u_g = gradients(u, x)[0]
        v_g = gradients(v, x)[0]
        p_g = gradients(p, x)[0]
        u_t, u_x, u_y = u_g[:,0], u_g[:,1], u_g[:,2]   
        v_t, v_x, v_y = v_g[:,0], v_g[:,1], v_g[:,2]
        p_x, p_y = p_g[:,1], p_g[:,2]

        u_gg = gradients(u_x, x)[0]
        v_gg = gradients(v_x, x)[0]
        u_xx, u_yy = u_gg[:,1], u_gg[:,2]
        v_xx, v_yy = v_gg[:,1], v_gg[:,2]

        loss_u = u_t + (u*u_x + v*u_y) + (p_x+self.P) - self.nu*(u_xx + u_yy)
        loss_v = v_t + (u*v_x + v*v_y) + p_y - self.nu*(v_xx + v_yy)
        loss_div_free = u_x + u_y
        return (loss_u**2).mean() + (loss_v**2).mean() + (loss_div_free**2).mean()
    
    def loss_ic(self, x, u):
        out = self.net(x)
        u_pred, v_pred = out[:,0], out[:,1]
        u_init, v_init = u[:,0], u[:,1]
        loss = ((u_pred-u_init)**2).mean() + ((v_pred-v_init)**2).mean()
        return loss

    def loss_bc(self, x, u):
        out = self.net(x)
        u_pred, v_pred = out[:,0], out[:,1]
        u_bound, v_bound = u[:,0], u[:,1]
        loss = ((u_pred-u_bound)**2).mean() + ((v_pred-v_bound)**2).mean()
        return loss

def bound_cond_lid(x_up, x_dw, x_l, x_r):
    ones = velocity*np.ones((x_up.shape[0], 1)) ## might have problem when num_x != num_y
    zeros = np.zeros((x_up.shape[0], 1))
    u_up, u_dw, u_l, u_r = ones, zeros, zeros, zeros
    v_up, v_dw, v_l, v_r = zeros, zeros, zeros, zeros

    return u_up, u_dw, u_l, u_r, v_up, v_dw, v_l, v_r

def init_cond_lid(x_init):
    u_init = np.zeros((x_init.shape[0]))
    v_init = np.zeros((x_init.shape[0]))
    return u_init, v_init

def main():
    ## parameters
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(device)

    epochs = 5000
    lr = 0.001
    
    ## boundary sampling points, inner sampling points, initial sampling points 
    # num_b_train, num_f_train, num_i_train = 400, 5000, 2500   
    x_min, x_max, y_min, y_max, t_min, t_max = 0, 2, 0, 2, 0, 2
    num_x, num_y, num_t = 25, 25, 10

    ## Generate data
    X = np.linspace(x_min, x_max, num=num_x)[:, None]
    Y = np.linspace(y_min, y_max, num=num_y)[:, None]
    T = np.linspace(t_min, t_max, num=num_t)[:, None]
    x_grid, y_grid = np.meshgrid(X, Y)
    x_grid_inner, y_grid_inner = x_grid[1:-1, 1:-1], y_grid[1:-1,1:-1]
    x_inner = np.concatenate((x_grid_inner.flatten()[:,None], y_grid_inner.flatten()[:,None]), axis=1)
    x = np.concatenate((x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)

    ## initial condition
    x_init = np.concatenate((np.zeros(( (num_x-2)*(num_y-2), 1 )), x_inner), axis=1)
    u_init, v_init = init_cond_lid(x_init)
    u_init = np.hstack((u_init[:,None], v_init[:,None]))

    ## boundary condition
    x_up = np.vstack((x_grid[-1,1:-1], y_grid[-1,1:-1])).T
    x_dw = np.vstack((x_grid[0,1:-1], y_grid[0,1:-1])).T
    x_l = np.vstack((x_grid[1:-1, 0], y_grid[1:-1, 0])).T
    x_r = np.vstack((x_grid[1:-1, -1], y_grid[1:-1, -1])).T
    u_up, u_dw, u_l, u_r, v_up, v_dw, v_l, v_r = bound_cond_lid(x_up, x_dw, x_l, x_r)

    x_up_ext = np.tile(x_up, (num_t,1))
    x_dw_ext = np.tile(x_dw, (num_t,1))
    x_l_ext = np.tile(x_l, (num_t,1))
    x_r_ext = np.tile(x_r, (num_t,1))

    u_up_bound = np.tile(u_up, (num_t, 1))
    u_dw_bound = np.tile(u_dw, (num_t, 1))
    u_l_bound = np.tile(u_l, (num_t, 1))
    u_r_bound = np.tile(u_r, (num_t, 1))
    v_up_bound = np.tile(v_up, (num_t, 1))
    v_dw_bound = np.tile(v_dw, (num_t, 1))
    v_l_bound = np.tile(v_l, (num_t, 1))
    v_r_bound = np.tile(v_r, (num_t, 1))
    
    u_up_bound = np.hstack((u_up_bound, v_up_bound))
    u_dw_bound = np.hstack((u_dw_bound, v_dw_bound))
    u_l_bound = np.hstack((u_l_bound, v_l_bound))
    u_r_bound = np.hstack((u_r_bound, v_r_bound))

    bsz = num_x-2 ## must be a sqaure region
    t_b_ext = np.zeros((num_t*bsz, 1))
    for i in range(0, num_t):
        t_b_ext[i*bsz:(i+1)*bsz,:] = T[i]

    x_up_bound = np.hstack((t_b_ext, x_up_ext))
    x_dw_bound = np.hstack((t_b_ext, x_dw_ext))
    x_l_bound = np.hstack((t_b_ext, x_l_ext))
    x_r_bound = np.hstack((t_b_ext, x_r_ext))
    
    x_bound = np.vstack((x_up_bound, x_dw_bound, x_l_bound, x_r_bound))
    u_bound = np.vstack((u_up_bound, u_dw_bound, u_l_bound, u_r_bound))

    ## inner points
    x_2d_ext = np.tile(x_inner, [num_t, 1])
    t_ext = np.zeros((num_t*(num_x-2)*(num_y-2), 1))
    for i in range (0, num_t):
        t_ext[i*(num_x-2)*(num_y-2):(i+1)*(num_x-2)*(num_y-2),:] = T[i]
    x_train = np.concatenate((t_ext, x_2d_ext), axis=1)

    ## sampling 
    # id_f = np.random.choice(num_x*num_y*num_t, num_f_train)
    # id_b = np.random.choice(2*(num_x+num_y)*num_t, num_b_train)
    # id_i = np.random.choice(num_x*num_y, num_i_train)

    # Transfer data to the device
    x_test_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32).to(device)
    x_train_tensor = torch.tensor(x_train, requires_grad=True, dtype=torch.float32).to(device)
    x_init_tensor = torch.tensor(x_init, requires_grad=True, dtype=torch.float32).to(device)
    u_init_tensor = torch.tensor(u_init, requires_grad=True, dtype=torch.float32).to(device)
    x_bound_tensor = torch.tensor(x_bound, requires_grad=True, dtype=torch.float32).to(device)
    u_bound_tensor = torch.tensor(u_bound, requires_grad=True, dtype=torch.float32).to(device)

    ## instantiate model and optimizer
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(x_train_tensor)
            loss_ic = model.loss_ic(x_init_tensor, u_init_tensor)
            loss_bc = model.loss_bc(x_bound_tensor, u_bound_tensor)
            loss = 5*loss_pde + loss_ic + 20*loss_bc
            print(f'epoch {epoch} loss_pde:{loss_pde:6f}, loss_bc:{loss_bc:6f}, loss_ic:{loss_ic:6f}')
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        # print(f'epoch {epoch}: loss {loss_value:.6f}')

    print('start training...')
    tic = time.time()
    for epoch in range(1, epochs + 1):    
        train(epoch)
    toc = time.time()
    print(f'total training time: {toc-tic}')

    ## plotting
    u_test = np.zeros((num_t, num_x, num_y))
    v_test = np.zeros((num_t, num_x, num_y))
    velo_test = np.zeros((num_t, num_x, num_y))
    T = np.linspace(t_min, t_max, num_t)
    for i in range(0, num_t):
        print(f'plotting snapshot: {i}')

        xt = np.concatenate((T[i]*np.ones((num_x*num_y, 1)), x), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.float32).to(device)
        y_pred = model(xt_tensor)
        u_test[i,:,:] = to_numpy(y_pred[:, 0]).reshape(num_x, num_y)
        v_test[i,:,:] = to_numpy(y_pred[:, 1]).reshape(num_x, num_y)
        velo_test[i,:,:] = np.sqrt(u_test[i]**2 + v_test[i]**2)

        fig, ax = newfig(2.0, 1.1)
        ax.axis('off') 
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(velo_test[i,:,:], interpolation='nearest', cmap='rainbow', origin='lower', aspect='auto')
        fig.colorbar(h)
        ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)
        line = np.linspace(xt.min(), xt.max(), 2)[:,None]
        fig.savefig('./velo_'+str(i+1000)+'.png')

        ## u
        fig, ax = newfig(2.0, 1.1)
        ax.axis('off') 
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(u_test[i,:,:], interpolation='nearest', cmap='rainbow', origin='lower', aspect='auto')
        fig.colorbar(h)
        ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)
        line = np.linspace(xt.min(), xt.max(), 2)[:,None]
        fig.savefig('./u_'+str(i+1000)+'.png')

        ## v
        fig, ax = newfig(2.0, 1.1)
        ax.axis('off') 
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(v_test[i,:,:], interpolation='nearest', cmap='rainbow', origin='lower', aspect='auto')
        fig.colorbar(h)
        ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)
        line = np.linspace(xt.min(), xt.max(), 2)[:,None]
        fig.savefig('./v_'+str(i+1000)+'.png')

if __name__ == '__main__':
    main()


















