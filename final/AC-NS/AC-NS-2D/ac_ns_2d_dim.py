"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn Navier-Stokes equation (2D).
    
    d(phi)/dt + div*(phi*u) = 1/Pe*(-F'(phi) + eta^2*lap(phi)) + beta(t)

    where:
    Input: [t, x, y]
    Output: [u, v, p, phi]
"""

import sys
sys.path.insert(0,'../../Utils')

import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io

from plotting import newfig, savefig

torch.manual_seed(123456)
np.random.seed(123456)

eta = 0.1
lmd = 0.01
gemma = 0.01
rho = 1
mu = 0.01

H_o = 0.1
k = 6*np.pi

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(3, 20))
        # self.net.add_module('tanh_layer_1', nn.Tanh())
        self.net.add_module('lrelu_layer_1', nn.LeakyReLU())
        for num in range(2,14):
            self.net.add_module('linear_layer_%d' %(num), nn.Linear(20, 20))
            # self.net.add_module('tanh_layer_%d' %(num), nn.Tanh())
            self.net.add_module('lrelu_layer_%d' %(num), nn.LeakyReLU())
        self.net.add_module('linear_layer_50', nn.Linear(20, 4))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        y = self.net(x)
        u, v, p, phi = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        u_g, v_g, p_g, phi_g = gradients(u, x)[0], gradients(v, x)[0], gradients(p, x)[0], gradients(phi, x)[0]
        u_t, u_x, u_y = u_g[:, 0], u_g[:, 1], u_g[:, 2]
        v_t, v_x, v_y = v_g[:, 0], v_g[:, 1], v_g[:, 2]
        p_x, p_y = p_g[:, 1], p_g[:, 2]
        phi_t, phi_x, phi_y = phi_g[:, 0], phi_g[:, 1], phi_g[:, 2]

        u_xx, u_yy = gradients(u_x, x)[0][:, 1], gradients(u_y, x)[0][:, 2]
        v_xx, v_yy = gradients(v_x, x)[0][:, 1], gradients(v_y, x)[0][:, 2]
        phi_xx, phi_xy, phi_yy = gradients(phi_x, x)[0][:, 1], gradients(phi_x, x)[0][:, 2], gradients(phi_y, x)[0][:, 2]

        ## NS
        res_ns_x = rho*(u_t + u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy) \
                    + lmd*(2*phi_x*phi_xx + phi_yy*phi_x + phi_xy*phi_y)
        res_ns_y = rho*(v_t + u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy) \
                    + lmd*(phi_xy*phi_x + phi_xx*phi_y + 2*phi_y*phi_yy)
        res_incomp = u_x + v_y

        ## PF
        F_g = phi**3 - phi
        res_pf = phi_t + eta**2*(u*phi_x + v*phi_y) - gemma*(-F_g + eta**2*(phi_xx + phi_yy)) 
        return (res_pf**2).mean() + (res_incomp**2).mean() + (res_ns_x**2).mean() + (res_ns_y**2).mean()

    def loss_bc(self, x_l, x_r, x_up, x_dw):
        y_l, y_r, y_up, y_dw = self.net(x_l), self.net(x_r), self.net(x_up), self.net(x_dw)

        u_l, v_l, p_l, phi_l = y_l[:, 0], y_l[:, 1], y_l[:, 2], y_l[:, 3]
        u_r, v_r, p_r, phi_r = y_r[:, 0], y_r[:, 1], y_r[:, 2], y_r[:, 3]
        u_up, v_up, phi_up = y_up[:, 0], y_up[:, 1], y_up[:, 3]
        u_dw, v_dw, phi_dw = y_dw[:, 0], y_dw[:, 1], y_dw[:, 3] 

        phi_up_y = gradients(phi_up, x_up)[0][:, 2]
        phi_dw_y = gradients(phi_dw, x_dw)[0][:, 2]
        return ((u_l - u_r)**2).mean() + ((v_l - v_r)**2).mean() +\
                ((p_l - p_r)**2).mean() + ((phi_l - phi_r)**2).mean() +\
                (u_up**2).mean() + (u_dw**2).mean() + (v_up**2).mean() + (v_dw**2).mean() +\
                (phi_dw_y**2).mean() + (phi_up_y**2).mean()

    def loss_ic(self, x_i, u_i, v_i, phi_i):
        y_pred = self.net(x_i)
        u_i_pred, v_i_pred, phi_i_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 3]
        # print(f'u_loss: {((u_i_pred-u_i)**2).mean():6f}')
        # print(f'v_loss: {((v_i_pred-v_i)**2).mean():6f}')
        # print(f'phi_loss: {((phi_i_pred-phi_i)**2).mean():6f}')
        return ((u_i_pred-u_i)**2).mean() + ((v_i_pred-v_i)**2).mean() + ((phi_i_pred-phi_i)**2).mean()

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def AC_2D_init(x):
    ## bubble
    u_init = np.zeros((x.shape[0]))
    v_init = np.zeros((x.shape[0]))
    phi_init = np.tanh( (0.2-np.sqrt((x[:,1]-0.0)**2 + (x[:,2] - 0.0)**2)) / (np.sqrt(2)*0.03) )
    return u_init, v_init, phi_init
    
    ## wave
    # u_init = np.zeros((x.shape[0]))
    # v_init = np.zeros((x.shape[0]))
    # phi_init = np.tanh((x[:, 2] - H_o*np.cos(k*x[:, 1]))/np.sqrt(2)/eta)
    # return u_init, v_init, phi_init

def main():
    ## parameters
    device = torch.device(f"cpu")
    epochs = 10000
    lr = 0.001

    num_x = 100
    num_y = 100
    num_t = 100
    num_b_train = 50    # boundary sampling points
    num_f_train = 50000   # inner sampling points
    num_i_train = 5000   # initial sampling points 
    num_t_train = 40

    x = np.linspace(-1, 1, num=num_x)
    y = np.linspace(-1, 1, num=num_y)
    t = np.linspace(0, 5, num=num_t)[:, None]
    x_grid, y_grid = np.meshgrid(x, y)
    # x_test = np.concatenate((t_grid.flatten()[:,None], x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    x_2d = np.concatenate((x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    
    ## initialization
    xt_init = np.concatenate((np.zeros((num_x*num_y, 1)), x_2d), axis=1)
    u_init, v_init, phi_init = AC_2D_init(xt_init)

    ## save init fig
    fig, ax = newfig(2.0, 1.1)
    ax.axis('off') 
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(phi_init.reshape(num_x, num_y), interpolation='nearest', cmap='rainbow', 
            # extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    fig.colorbar(h)
    ax.plot(xt_init[:,1], xt_init[:,2], 'kx', label = 'Data (%d points)' % (xt_init.shape[0]), markersize = 4, clip_on = False)
    line = np.linspace(xt_init.min(), xt_init.max(), 2)[:,None]
    fig.savefig('./phi_init.png')


    x_2d_ext = np.tile(x_2d, [num_t,1])
    t_ext = np.zeros((num_t*num_x*num_y, 1))
    for i in range (0, num_t):
        t_ext[i*num_x*num_y:(i+1)*num_x*num_y,:] = t[i]
    xt_2d_ext = np.concatenate((t_ext, x_2d_ext), axis=1)
    
    ## sampling
    id_f = np.random.choice(num_x*num_y*num_t, num_f_train, replace=False)
    id_b = np.random.choice(num_x, num_b_train, replace=False) ## Dirichlet
    id_i = np.random.choice(num_x*num_y, num_i_train, replace=False)
    id_t = np.random.choice(num_t, num_t_train, replace=False)

    ## boundary  
    t_b = t[id_t,:]
    t_b_ext = np.zeros((num_t_train*num_b_train, 1))
    for i in range(0, num_t_train):
        t_b_ext[i*num_b_train:(i+1)*num_b_train,:] = t_b[i]
    x_up = np.vstack((x_grid[-1,:], y_grid[-1,:])).T
    x_dw = np.vstack((x_grid[0,:], y_grid[0,:])).T
    x_l = np.vstack((x_grid[:, 0], y_grid[:, 0])).T
    x_r = np.vstack((x_grid[:, -1], y_grid[:, -1])).T
    # x_bound = np.vstack((x_up, x_dw, x_l, x_r))

    x_up_sample = x_up[id_b, :]
    x_dw_sample = x_dw[id_b, :]
    x_l_sample = x_l[id_b, :]
    x_r_sample = x_r[id_b, :]

    x_up_ext = np.tile(x_up_sample, (num_t_train, 1))
    x_dw_ext = np.tile(x_dw_sample, (num_t_train, 1))
    x_l_ext = np.tile(x_l_sample, (num_t_train, 1))
    x_r_ext = np.tile(x_r_sample, (num_t_train, 1))

    xt_up = np.hstack((t_b_ext, x_up_ext))
    xt_dw = np.hstack((t_b_ext, x_dw_ext))
    xt_l = np.hstack((t_b_ext, x_l_ext))
    xt_r = np.hstack((t_b_ext, x_r_ext))

    xt_i = xt_init[id_i, :]
    xt_f = xt_2d_ext[id_f, :]

    ## set data as tensor and send to device
    xt_f_train = torch.tensor(xt_f, requires_grad=True, dtype=torch.float32).to(device)
    # x_test = torch.tensor(xt_2d_ext, requires_grad=True, dtype=torch.float32).to(device)
    xt_i_train = torch.tensor(xt_init, requires_grad=True, dtype=torch.float32).to(device)
    x_i_train = torch.tensor(x_2d, requires_grad=True, dtype=torch.float32).to(device)

    u_i_train = torch.tensor(u_init, dtype=torch.float32).to(device)
    v_i_train = torch.tensor(v_init, dtype=torch.float32).to(device)
    phi_i_train = torch.tensor(phi_init, dtype=torch.float32).to(device)
    xt_l_train = torch.tensor(xt_l, requires_grad=True, dtype=torch.float32).to(device)
    xt_r_train = torch.tensor(xt_r, requires_grad=True, dtype=torch.float32).to(device)
    xt_up_train = torch.tensor(xt_up, requires_grad=True, dtype=torch.float32).to(device)
    xt_dw_train = torch.tensor(xt_dw, requires_grad=True, dtype=torch.float32).to(device)

    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(xt_f_train)
            loss_bc = model.loss_bc(xt_l_train, xt_r_train, xt_up_train, xt_dw_train)
            loss_ic = model.loss_ic(xt_i_train, u_i_train, v_i_train, phi_i_train)
            loss = loss_pde + loss_bc + 100*loss_ic
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

    ## test
    phi_test = np.zeros((num_t, num_x, num_y))
    for i in range(0, num_t):
        xt = np.concatenate((t[i]*np.ones((num_x*num_y, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.float32).to(device)
        y_pred = model(xt_tensor)
        phi_test[i,:,:] = to_numpy(y_pred[:, 3]).reshape(num_x, num_y)

        fig, ax = newfig(2.0, 1.1)
        ax.axis('off') 
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(phi_test[i,:,:], interpolation='nearest', cmap='rainbow', 
                # extent=[t.min(), t.max(), x.min(), x.max()], 
                    origin='lower', aspect='auto')
        fig.colorbar(h)
        ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)
        line = np.linspace(xt.min(), xt.max(), 2)[:,None]
        fig.savefig('./phi_'+str(i+1000)+'.png')

if __name__ == '__main__':
    main()


















