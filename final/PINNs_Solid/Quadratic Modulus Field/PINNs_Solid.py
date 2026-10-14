# Framework based on https://arxiv.org/abs/1711.10561
# Params: 80-20-0.2, 1+X^2-Y^2

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
#from pyDOE import lhs
#from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, layers_U, layers_E, xnu, xb, yb, F_val, f, num_train_tps, num_test_tps):

        # Initialize NNs
        self.layers_U = layers_U
        self.weights_U, self.biases_U, self.adaps_U = self.initialize_NN(layers_U)
        self.layers_E = layers_E
        self.weights_E, self.biases_E, self.adaps_E = self.initialize_NN(layers_E)

        # Parameters
        self.xnu = xnu
        self.xb = xb
        self.yb = yb
        self.lb = np.array([xb[0], yb[0]])
        self.ub = np.array([xb[1], yb[1]])

        # Output file
        self.f = f
        
        # tf Placeholders
        # Uninformed: Rigid Body Points
        self.x1_tf = tf.placeholder(tf.float32, shape = [1, 1])
        self.x2_tf = tf.placeholder(tf.float32, shape = [1, 1])
        self.y1_tf = tf.placeholder(tf.float32, shape = [1, 1])
        self.y2_tf = tf.placeholder(tf.float32, shape = [1, 1])

        # Informed
        self.xx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.yx_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # Training Data
        self.ux_star_tf = tf.placeholder(tf.float32, shape = [None, 1])
        self.vx_star_tf = tf.placeholder(tf.float32, shape = [None, 1])

        # Test Location
        self.x_test_tf = tf.placeholder(tf.float32, shape = [None, 1])
        self.y_test_tf = tf.placeholder(tf.float32, shape = [None, 1])

        # Generate Training and Testing Points
        self.generateTrain(num_train_tps)
        self.generateTest(num_test_tps)

        # tf Graphs
        # Uninformed
        # Interior
        # Original (same as part 1)
        self.f1, self.f2, self.uI, self.vI, self.sxxI, self.sxyI, self.syyI, _ = self.net_f(self.xI, self.yI)
        # Left BC
        _, _, self.uL, self.vL, self.sxxL, self.sxyL, self.syyL, _ = self.net_f(self.xL, self.yL)
        # Right BC
        _, _, self.uR, self.vR, self.sxxR, self.sxyR, self.syyR, _ = self.net_f(self.xR, self.yR)
        # Top (Upper) BC
        _, _, self.uU, self.vU, self.sxxU, self.sxyU, self.syyU, _ = self.net_f(self.xU, self.yU)
        # Bottom (Lower) BC
        _, _, self.uD, self.vD, self.sxxD, self.sxyD, self.syyD, _ = self.net_f(self.xD, self.yD)
        # RBP
        _, _, self.u1, self.v1, _, _, _, _ = self.net_f(self.x1_tf, self.y1_tf)
        _, _, self.u2, self.v2, _, _, _, _ = self.net_f(self.x2_tf, self.y2_tf)
        # Informed
        _, _, self.ux, self.vx, self.sxxx, self.sxyx, self.syyx, _ = self.net_f(self.xx_tf, self.yx_tf)

        # Test
        self.f1_test, self.f2_test, self.u_test, self.v_test, self.sxx_test, self.sxy_test, self.syy_test, self.E_test = self.net_f(self.x_test_tf, self.y_test_tf)

        # Loss
        # Interior
        self.loss_I =  (tf.reduce_mean(tf.square(self.f1)) + tf.reduce_mean(tf.square(self.f2)))
        # Left & Right: tension
        self.loss_L = tf.reduce_mean(tf.square(self.sxxL - F_val)) + tf.reduce_mean(tf.square(self.sxyL))
        self.loss_R = tf.reduce_mean(tf.square(self.sxxR - F_val)) + tf.reduce_mean(tf.square(self.sxyR))
        # Up & Down: free
        self.loss_U = tf.reduce_mean(tf.square(self.sxyU)) + tf.reduce_mean(tf.square(self.syyU))
        self.loss_D = tf.reduce_mean(tf.square(self.sxyD)) + tf.reduce_mean(tf.square(self.syyD))
        # Rigid Body Motion
        self.loss_rigid = tf.square(tf.reduce_sum(self.u1)) + tf.square(tf.reduce_sum(self.v1)) + tf.square(tf.reduce_sum(self.v2))
        # Informed points
        self.loss_x = tf.reduce_mean(tf.square(self.ux - self.ux_star_tf)) + tf.reduce_sum(tf.square(self.vx - self.vx_star_tf))
        # Total Loss
        self.loss = self.loss_I * 1.0 + (self.loss_L + self.loss_R) * 1.0 + self.loss_U + self.loss_D + self.loss_rigid * 1.0  + self.loss_x * 1.0 

        # Optimizer
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        adaps = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            a = tf.Variable(1.0, dtype=tf.float32)
            weights.append(W)
            biases.append(b)
            adaps.append(a)        
        return weights, biases, adaps
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def regularize(self):
        weights = self.weights
        loss_reg = 0
        for i in range(len(weights)):
            loss_reg = loss_reg + tf.reduce_mean(tf.square(weights[i]))
        return loss_reg

    def neural_net(self, X):
        weights = self.weights_U
        biases = self.biases_U
        adaps = self.adaps_U
        num_layers = len(weights) + 1
        h = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a,tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        U = tf.add(tf.matmul(h, W), b)
        return U

    def net_E(self, X):
        weights = self.weights_E
        biases = self.biases_E
        adaps = self.adaps_E
        num_layers = len(weights) + 1
        h = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a,tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        E = tf.add(tf.matmul(h, W), b)
        return E + 1.0      # shift towards positive values

    def net_f(self, x, y):
        X = tf.concat([x, y], 1)
        U = self.neural_net(X)
        u = U[:, 0:1]
        v = U[:, 1:2]
        ENN = self.net_E(X)
        mu = ENN / (2.0 * (1 + self.xnu))
        lambd = self.xnu * ENN / (1 + self.xnu) / (1 - 2 * self.xnu)
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        exx = u_x
        exy = 0.5 * (u_y + v_x)
        eyy = v_y
        tre = exx + eyy
        sxx = lambd * tre + 2 * mu * exx
        sxy = 2 * mu * exy
        syy = lambd * tre + 2 * mu * eyy
        sxx_x = tf.gradients(sxx, x)[0]
        sxx_y = tf.gradients(sxx, y)[0]
        sxy_x = tf.gradients(sxy, x)[0]
        sxy_y = tf.gradients(sxy, y)[0]
        syy_x = tf.gradients(syy, x)[0]
        syy_y = tf.gradients(syy, y)[0]
        f1 = sxx_x + sxy_y
        f2 = sxy_x + syy_y
        return f1, f2, u, v, sxx, sxy, syy, ENN

    def generateTrain(self, num_line):
        x_1D = tf.linspace(np.float32(self.xb[0]), np.float32(self.xb[1]), num_line+1)
        y_1D = tf.linspace(np.float32(self.yb[0]), np.float32(self.yb[1]), num_line+1)
        x_2D = tf.multiply(x_1D[:,None], tf.ones([1, num_line+1]))
        y_2D = tf.multiply(tf.ones([num_line+1, 1]), y_1D[None,:])
        self.xI = tf.reshape(x_2D, [-1, 1])
        self.yI = tf.reshape(y_2D, [-1, 1])

        # Top boundary
        msize = (self.xb[1] - self.xb[0]) / num_line
        x_outer = tf.concat([[np.float32(self.xb[1])], tf.linspace(np.float32(self.xb[1] - msize/2), np.float32(self.xb[0] + msize/2) , num_line), [np.float32(self.xb[0])]], axis = 0)
        y_outer = self.yb[1] * tf.ones(x_outer.shape)
        self.xU = x_outer[:, None]
        self.yU = y_outer[:, None]

        # Left boundary
        msize = (self.yb[1] - self.yb[0]) / num_line
        y_outer = tf.concat([[np.float32(self.yb[1])], tf.linspace(np.float32(self.yb[1] - msize/2), np.float32(self.yb[0] + msize/2), num_line), [np.float32(self.yb[0])]], axis = 0)
        x_outer = self.xb[0] * tf.ones(y_outer.shape)    
        self.xL = x_outer[:, None]
        self.yL = y_outer[:, None]

        # Bottom boundary
        msize = (self.xb[1] - self.xb[0]) / num_line
        x_outer = tf.concat([[np.float32(self.xb[0])], tf.linspace(np.float32(self.xb[0] + msize/2), np.float32(self.xb[1] - msize/2) , num_line), [np.float32(self.xb[1])]], axis = 0)
        y_outer = self.yb[0] * tf.ones(x_outer.shape)
        self.xD = x_outer[:, None]
        self.yD = y_outer[:, None]

        # Right boundary
        msize = (self.yb[1] - self.yb[0]) / num_line  
        y_outer = tf.concat([[np.float32(self.yb[0])], tf.linspace(np.float32(self.yb[0] + msize/2), np.float32(self.yb[1] - msize/2) , num_line), [np.float32(self.yb[1])]], axis = 0)
        x_outer = self.xb[1] * tf.ones(y_outer.shape)
        self.xR = x_outer[:, None]
        self.yR = y_outer[:, None]
        return

    def generateTest(self, num_line):
        x_1D = np.linspace(self.xb[0], self.xb[1], num_line+1)
        y_1D = np.linspace(self.yb[0], self.yb[1], num_line+1)
        x_2D = np.matmul(x_1D[:,None], np.ones((1, num_line+1)))
        y_2D = np.matmul(np.ones((num_line+1, 1)), y_1D[None,:])
        self.x_test = np.reshape(x_2D, [-1, 1])
        self.y_test = np.reshape(y_2D, [-1, 1])
        return

    def train(self, Xx, Ux, it, printloss):

        xx = Xx[:, 0:1]
        yx = Xx[:, 1:2]
        ux = Ux[:, 0:1]
        vx = Ux[:, 1:2]

        # rigid body points
        x1 = np.array([[self.xb[0]]])
        x2 = np.array([[self.xb[1]]])
        y1 = np.array([[self.yb[0]]])
        y2 = np.array([[self.yb[0]]])


 
        tf_dict = { self.x1_tf: x1, self.y1_tf: y1,
                    self.x2_tf: x2, self.y2_tf: y2,
                    self.xx_tf: xx, self.yx_tf: yx,
                    self.ux_star_tf: ux, self.vx_star_tf: vx,
                    }
        self.sess.run(self.train_op_Adam, tf_dict)

        # Print
        if (printloss):    
            loss_value = self.sess.run(self.loss, tf_dict)
            loss_value_I = self.sess.run(self.loss_I, tf_dict)
            loss_value_L = self.sess.run(self.loss_L, tf_dict)
            loss_value_R = self.sess.run(self.loss_R, tf_dict)
            loss_value_U = self.sess.run(self.loss_U, tf_dict)
            loss_value_D = self.sess.run(self.loss_D, tf_dict)
            loss_value_rigid = self.sess.run(self.loss_rigid, tf_dict)
            loss_value_x = self.sess.run(self.loss_x, tf_dict)
            loss_value_array = [loss_value_I, loss_value_L, loss_value_R, loss_value_U, loss_value_D, loss_value_rigid, loss_value_x]
            np.set_printoptions(precision=3)
            content = 'It: %d, Loss: %.3e' %(it, loss_value) +  '  Losses ILRUDrxIBIC:' + str(loss_value_array)
            print(content)
            self.f.write(content + "\n")
            return loss_value
        else:
            return None
        

    def test(self):
        tf_dict = {self.x_test_tf: self.x_test, self.y_test_tf: self.y_test}
        u_test = self.sess.run(self.u_test, tf_dict)
        v_test = self.sess.run(self.v_test, tf_dict)
        #sxx_test = self.sess.run(self.sxx_test, tf_dict)
        #sxy_test = self.sess.run(self.sxy_test, tf_dict)
        #syy_test = self.sess.run(self.syy_test, tf_dict)
        #f1_test = self.sess.run(self.f1_test, tf_dict)
        #f2_test = self.sess.run(self.f2_test, tf_dict)
        E_test = self.sess.run(self.E_test, tf_dict)
        X_test = np.hstack((self.x_test, self.y_test))
        U_test = np.hstack((u_test, v_test))
        return X_test, U_test, E_test

def import_data(filepath):
    data = np.load(filepath, encoding = 'latin1')
    Xx = data[0][:, 0:2]
    Ux = data[0][:, 2:]
    XFEM = data[1][:,1:3]
    UFEM = data[1][:,3:]
    return Xx, Ux, XFEM, UFEM

def problem_name(id):
    if (id==1):
        # Linear Case
        name = 'Linear'
    elif (id==2):
        # Quadratic Case
        name = 'Quadratic'
    elif (id==3):
        # Sinusoidal Case
        name = 'Sinusoidal'
    return name

def problem_E(x, y, id):
    if (id==1):
        # Linear Case
        E = x + 1.0
    elif (id==2):
        # Quadratic Case
        E = 1.0 + x ** 2 - y ** 2
    elif (id==3):
        # Sinusoidal Case
        E = 1.0 + 0.2 * np.sin(2*np.pi*x) + 0.2 * np.sin(3*np.pi*y)
    return E

def DoubleNN(nIter = 1000000, print_period = 1000, plot_period = 10000):

    # Geometry (domain bounds) and Meshing
    xb = np.array([-0.5, 0.5])
    yb = np.array([-0.5, 0.5])

    # Material properties
    xnu = 0.2

    # load value
    F_val = 1.0

    # Network Structure
    layers_U = [2, 30, 30, 30, 30, 3]
    layers_E = [2, 30, 30, 30, 30, 1]

    problem_id = 2

    filename = problem_name(problem_id)

    # Generate training (informed) & true value
    Xx, Ux, XFEM, UFEM = import_data('Output_' + filename + '.npy')

    f = open("loss_record.txt","w")

    num_train_tps = 40
    num_test_tps = 80

    # Create the model
    model = PhysicsInformedNN(layers_U, layers_E, xnu, xb, yb, F_val, f, num_train_tps, num_test_tps)

    it_array = []
    loss_array = []
    data_save = {}

    start_time = time.time()

    for it in range(1, nIter+1):
        loss = model.train(Xx, Ux, it, it%print_period==0)
        if (it%print_period==0):
            loss_array.append(loss)
            it_array.append(it)
            dt = time.time() - start_time
            print('Time: ', dt)
            start_time = time.time()
        if (it%plot_period==0):
            X_test, U_test, E_test = model.test()
            print("Result Plotted...")
            #plt.rcParams.update({'font.size': 14})
            #################### Deformation ####################
            plt.figure(1)
            figtopic = 'Displacement'
            ampli = 0.5
            x0 = X_test[:,0]+0*U_test[:,0]*ampli
            y0 = X_test[:,1]+0*U_test[:,1]*ampli
            xPINNs = X_test[:,0]+U_test[:,0]*ampli
            yPINNs = X_test[:,1]+U_test[:,1]*ampli
            xAbaqus = XFEM[:,0]+UFEM[:,0]*ampli
            yAbaqus = XFEM[:,1]+UFEM[:,1]*ampli
            plt.plot(x0, y0, '.k' , label = 'undeformed')
            plt.plot(xAbaqus, yAbaqus, 'xb', label = 'Abaqus')
            plt.plot(xPINNs, yPINNs, '.r' , label = 'deformed')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.legend()
            plt.title(figtopic)
            plt.axis('equal')
            plt.xlim((-0.6, 1.2))
            plt.ylim((-0.6, 0.6))
            plt.savefig('Fig_' + figtopic + '_It_' + str(it) + '.png')

            data_fig1 = (x0, y0, xPINNs, yPINNs, xAbaqus, yAbaqus)

            plt.close(1)

            #################### E Profile ####################
            plt.figure(2)
            figtopic = 'Modulus'
            x2D = np.reshape(X_test[:,0:1], [num_test_tps+1, num_test_tps+1])
            y2D = np.reshape(X_test[:,1:], [num_test_tps+1, num_test_tps+1])
            E2D_PINNs = np.reshape(E_test, [num_test_tps+1, num_test_tps+1])
            E2D_Abaqus = np.reshape(problem_E(x2D, y2D, problem_id), [num_test_tps+1, num_test_tps+1])
            cs = plt.contourf(x2D, y2D, E2D_PINNs, 100)
            plt.colorbar(cs)
            #plt.clim(-0.2, 2.0)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig('Fig_' + figtopic + '_It_' + str(it) + '.png')

            plt.close(2)

            #################### Err_E Profile ####################
            plt.figure(3)
            figtopic = 'ModulusError'
            cs = plt.contourf(x2D, y2D, E2D_PINNs-E2D_Abaqus, 100)
            plt.colorbar(cs)
            #plt.clim(-0.2, 2.0)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig('Fig_' + figtopic + '_It_' + str(it) + '.png')

            data_fig23 = (x2D, y2D, E2D_PINNs, E2D_Abaqus)

            plt.close(3)

            #################### loss ####################
            plt.figure(4)
            figtopic = 'loss'
            plt.semilogy(it_array, loss_array, '-b')
            plt.xlabel("Num Iteration")
            plt.ylabel("Loss")
            plt.title(figtopic)
            plt.savefig('Fig_' + figtopic + '_It_' + str(it) + '.png')

            data_fig4 = (it_array, loss_array)

            plt.close(4)

            ################### Training Points #################
            plt.figure(5)
            figtopic = 'Training Points'
            xI = model.sess.run(model.xI)
            yI = model.sess.run(model.yI)
            xL = model.sess.run(model.xL)
            yL = model.sess.run(model.yL)
            xR = model.sess.run(model.xR)
            yR = model.sess.run(model.yR)
            xU = model.sess.run(model.xU)
            yU = model.sess.run(model.yU)
            xD = model.sess.run(model.xD)
            yD = model.sess.run(model.yD)
            xB = np.vstack((xL, xR, xU, xD))
            yB = np.vstack((yL, yR, yU, yD))
            xx = Xx[:,0:1]
            yx = Xx[:,1:2]
            plt.plot(xI, yI,'.k', label = 'PDE')
            plt.plot(xB, yB, '.b', label = 'Boundary')
            plt.plot(xx, yx, 'xr', label = 'Data')
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig('Fig_' + figtopic + '_It_' + str(it) + '.png')

            data_fig5 = (xI, yI, xB, yB, xx, yx)

            plt.close(5)

            if (it==plot_period):
                #################### E True Profile ####################
                plt.figure(6)
                figtopic = 'ModulusTrue'
                cs = plt.contourf(x2D, y2D, E2D_Abaqus, 100)
                plt.colorbar(cs)
                #plt.clim(-0.2, 2.0)
                plt.title(figtopic)
                plt.axis('equal')
                plt.savefig('Fig_' + figtopic + '_It_' + str(it) + '.png')

                plt.close(6)

            data_save[it] = (data_fig1, data_fig23, data_fig4, data_fig5)
            np.save('PINNs_Output.npy', data_save)

DoubleNN(nIter = 1000000, print_period = 1000, plot_period = 10000)