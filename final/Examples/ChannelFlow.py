import tensorflow as tf
import numpy as np
import scipy.io
import time
import horovod.tensorflow as hvd


# Hyper parameters
tor = 1e-4
batch_num = 150
lambda_b = 10.
lambda_i = 50.
layers = [4] + 10 * [6 * 50] + [4]

#
N_Skip = batch_num

class ChannelFlow:
    # Initialize the class
    def __init__(self, t, x, y, z,  # inside points
                 data_boundary,  # boundary conditions
                 data_initial,  # initial conditions
                 layers, Xmin, Xmax,   # network architecture
                 P_x, niu):  # flow parameters

        self.P_x = P_x
        self.niu = niu

        # data of boundary conditions
        self.tb = data_boundary[:, 0:1]
        self.xb = data_boundary[:, 1:2]
        self.yb = data_boundary[:, 2:3]
        self.zb = data_boundary[:, 3:4]
        self.ub = data_boundary[:, 4:5]
        self.vb = data_boundary[:, 5:6]
        self.wb = data_boundary[:, 6:7]
        self.pb = data_boundary[:, 7:8]

        # Number of data points for the boundary
        self.Nb = self.tb.shape[0]

        # data of initial conditions
        self.ti = data_initial[:, 0:1]
        self.xi = data_initial[:, 1:2]
        self.yi = data_initial[:, 2:3]
        self.zi = data_initial[:, 3:4]
        self.ui = data_initial[:, 4:5]
        self.vi = data_initial[:, 5:6]
        self.wi = data_initial[:, 6:7]
        self.pi = data_initial[:, 7:8]

        # Number of data points for the initial
        self.Ni = self.ti.shape[0]

        Xb = np.concatenate([self.tb, self.xb, self.yb, self.zb], 1)
        Xi = np.concatenate([self.ti, self.xi, self.yi, self.zi], 1)
        X = np.concatenate([Xb, Xi], 0)
        self.lower_X = Xmin #X.min(0)  # minimum of X
        self.upper_X = Xmax #X.max(0)  # maximum of X

        # data inside the domain
        self.te = t
        self.xe = x
        self.ye = y
        self.ze = z

        # Number of data points for the equation
        self.Ne = self.te.shape[0]

        # layers
        self.layers = layers

        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        
        
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.gpu_options.visible_device_list=str(hvd.local_rank())
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        self.learning_rate = tf.placeholder(tf.float32, shape=[])  # it can change during the training process

        # placeholders for data of boundary
        self.tb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.xb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.yb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.zb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.vb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.wb_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # placeholders for data of initial
        self.ti_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.xi_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.yi_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.zi_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ui_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.vi_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.wi_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # placeholders for data inside the domain
        self.te_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.xe_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ye_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ze_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # physics informed neural networks (boundary)
        (self.ub_pred, self.vb_pred, self.wb_pred, self.pb_pred,
         self.eq1b_pred, self.eq2b_pred, self.eq3b_pred, self.eq4b_pred) =\
            self.net_NS_AD(self.tb_tf, self.xb_tf, self.yb_tf, self.zb_tf)
        
        # physics informed neural networks (initial)
        (self.ui_pred, self.vi_pred, self.wi_pred, self.pi_pred,
         self.eq1i_pred, self.eq2i_pred, self.eq3i_pred, self.eq4i_pred) =\
            self.net_NS_AD(self.ti_tf, self.xi_tf, self.yi_tf, self.zi_tf)

        # physics informed neural networks (inside the domain)
        (self.u_pred, self.v_pred, self.w_pred, self.p_pred,
         self.eq1_pred, self.eq2_pred, self.eq3_pred, self.eq4_pred) =\
            self.net_NS_AD(self.te_tf, self.xe_tf, self.ye_tf, self.ze_tf)

        self.loss_b = tf.reduce_mean(tf.square(self.ub_tf - self.ub_pred)) + \
                      tf.reduce_mean(tf.square(self.vb_tf - self.vb_pred)) + \
                      tf.reduce_mean(tf.square(self.wb_tf - self.wb_pred))
        
        self.loss_i = tf.reduce_mean(tf.square(self.ui_tf - self.ui_pred)) + \
                      tf.reduce_mean(tf.square(self.vi_tf - self.vi_pred)) + \
                      tf.reduce_mean(tf.square(self.wi_tf - self.wi_pred))
        
        self.loss_eq1 = tf.reduce_mean(tf.square(self.eq1_pred)) + \
                        tf.reduce_mean(tf.square(self.eq1b_pred)) + \
                        tf.reduce_mean(tf.square(self.eq1i_pred))
        self.loss_eq2 = tf.reduce_mean(tf.square(self.eq2_pred)) + \
                        tf.reduce_mean(tf.square(self.eq2b_pred)) + \
                        tf.reduce_mean(tf.square(self.eq2i_pred))
        self.loss_eq3 = tf.reduce_mean(tf.square(self.eq3_pred)) + \
                        tf.reduce_mean(tf.square(self.eq3b_pred)) + \
                        tf.reduce_mean(tf.square(self.eq3i_pred))
        self.loss_eq4 = tf.reduce_mean(tf.square(self.eq4_pred)) + \
                        tf.reduce_mean(tf.square(self.eq4b_pred)) + \
                        tf.reduce_mean(tf.square(self.eq4i_pred))

        # loss
        self.loss = self.loss_b * lambda_b + self.loss_i * lambda_i + \
                    self.loss_eq1 + self.loss_eq2 + self.loss_eq3 + self.loss_eq4


        self.loss1=hvd.allreduce(self.loss)
        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer = hvd.DistributedOptimizer(self.optimizer)
        self.train_op = self.optimizer.minimize(self.loss)
        self.IteNum = 0
        self.epoch = 0
        self.lossCol = []
        init = tf.global_variables_initializer()
        bcast=hvd.broadcast_global_variables(0)
        self.sess.run(init)
        self.sess.run(bcast)
        #print("Weights", self.sess.run(self.weights), "Biases", self.sess.run(self.biases), "rank", hvd.rank() )

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases  # they are all three dimensional

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))  # stddev decreases with the increasing of in_dim + out_dim
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lower_X) / (self.upper_X - self.lower_X) - 1.0  # The input is normalize to [-1, 1].
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS_AD(self, t, x, y, z):
        uvwp = self.neural_net(tf.concat([t, x, y, z], 1), self.weights, self.biases)

        u = uvwp[:, 0:1]
        v = uvwp[:, 1:2]
        w = uvwp[:, 2:3]
        p = uvwp[:, 3:4]

        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]
        w_t = tf.gradients(w, t)[0]

        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]
        w_x = tf.gradients(w, x)[0]
        p_x = tf.gradients(p, x)[0]

        u_y = tf.gradients(u, y)[0]
        v_y = tf.gradients(v, y)[0]
        w_y = tf.gradients(w, y)[0]
        p_y = tf.gradients(p, y)[0]

        u_z = tf.gradients(u, z)[0]
        v_z = tf.gradients(v, z)[0]
        w_z = tf.gradients(w, z)[0]
        p_z = tf.gradients(p, z)[0]

        u_xx = tf.gradients(u_x, x)[0]
        v_xx = tf.gradients(v_x, x)[0]
        w_xx = tf.gradients(w_x, x)[0]

        u_yy = tf.gradients(u_y, y)[0]
        v_yy = tf.gradients(v_y, y)[0]
        w_yy = tf.gradients(w_y, y)[0]

        u_zz = tf.gradients(u_z, z)[0]
        v_zz = tf.gradients(v_z, z)[0]
        w_zz = tf.gradients(w_z, z)[0]

        eq1 = u_t + (u * u_x + v * u_y + w * u_z) + (p_x + self.P_x) - self.niu * (u_xx + u_yy + u_zz)
        eq2 = v_t + (u * v_x + v * v_y + w * v_z) + p_y - self.niu * (v_xx + v_yy + v_zz)
        eq3 = w_t + (u * w_x + v * w_y + w * w_z) + p_z - self.niu * (w_xx + w_yy + w_zz)
        eq4 = u_x + v_y + w_z

        return u, v, w, p, eq1, eq2, eq3, eq4

    def train(self, num_epochs, batch_number, learning_rate):
        batch_size_b = self.Nb // batch_number
        batch_size_i = self.Ni // batch_number
        batch_size_e = self.Ne // batch_number
        print("batch_size_b is ", batch_size_b)
        print("batch_size_i is ", batch_size_i)
        print("batch_size_e is ", batch_size_e)

        start_time = time.time()
        loss_value = 1.
        for epoch in range(num_epochs):
            if loss_value > tor:
                self.epoch += 1
                perm_b = np.random.permutation(self.Nb)
                perm_i = np.random.permutation(self.Ni)
                perm_e = np.random.permutation(self.Ne)
                for it in range(0, batch_num):
                    self.IteNum += 1
                    idx_b = perm_b[it*batch_size_b:(it+1)*batch_size_b]
                    idx_i = perm_i[it*batch_size_i:(it+1)*batch_size_i]
                    idx_e = perm_e[it*batch_size_e:(it+1)*batch_size_e]
    
                    tf_dict = {self.ti_tf: self.ti[idx_i, :], self.xi_tf: self.xi[idx_i, :], 
                               self.yi_tf: self.yi[idx_i, :], self.zi_tf: self.zi[idx_i, :],
                               self.ui_tf: self.ui[idx_i, :], self.vi_tf: self.vi[idx_i, :], 
                               self.wi_tf: self.wi[idx_i, :],  # initial
                               self.tb_tf: self.tb[idx_b, :], self.xb_tf: self.xb[idx_b, :],
                               self.yb_tf: self.yb[idx_b, :], self.zb_tf: self.zb[idx_b, :],
                               self.ub_tf: self.ub[idx_b, :], self.vb_tf: self.vb[idx_b, :],
                               self.wb_tf: self.wb[idx_b, :],  # boundary
                               self.te_tf: self.te[idx_e, :], self.xe_tf: self.xe[idx_e, :],
                               self.ye_tf: self.ye[idx_e, :], self.ze_tf: self.ze[idx_e, :],
                               self.learning_rate: learning_rate}
                    # print(tf_dict)
    
                    self.sess.run(self.train_op, tf_dict)
                    # Print
                    if self.IteNum % N_Skip == 0:
                        elapsed = time.time() - start_time
                        (loss_value,
                         loss_b, loss_i,
                         loss_eq1, loss_eq2, loss_eq3, loss_eq4,
                         learning_rate_value) = self.sess.run([self.loss1, self.loss_b, self.loss_i,
                                                               self.loss_eq1, self.loss_eq2,
                                                               self.loss_eq3, self.loss_eq4,
                                                               self.learning_rate], tf_dict)
                        print('***************************************************************')
                        print('Epoch: %d, It: %d, Time: %.2f, Learning Rate: %.1e, Loss: %.3e'
                              % (self.epoch, it, elapsed, learning_rate_value, loss_value))
                        print('LossI: %.3e, LossB: %.3e, LossEqX: %.3e, LossEqY: %.3e, LossEqZ: %.3e, LossEqIncom: %.3e'
                              % (loss_i, loss_b, loss_eq1, loss_eq2, loss_eq3, loss_eq4))
                        print('')
                        start_time = time.time()
                        self.lossCol.append([self.IteNum, loss_value, loss_b,
                                             loss_i, loss_eq1, loss_eq2, loss_eq3, loss_eq4])
        return loss_value

    
    
    def predict(self, t_star, x_star, y_star, z_star):

        tf_dict = {self.te_tf: t_star, self.xe_tf: x_star, self.ye_tf: y_star, self.ze_tf: z_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, w_star, p_star


# -------------------------------------------------------- #
# ------------------------Training------------------------ #
# -------------------------------------------------------- #
# Load Data
data = scipy.io.loadmat('./Data/Channel_1000plus.mat')

t_star = data['t_star']  # T x 1
x_star = data['x_star']  # N x 1
y_star = data['y_star']  # N x 1
z_star = data['z_star']  # N x 1

T = t_star.shape[0]
N = x_star.shape[0]

# Rearrange Data
T_star = np.tile(t_star, (1, N)).T  # N x T
X_star = np.tile(x_star, (1, T))  # N x T
Y_star = np.tile(y_star, (1, T))  # N x T
Z_star = np.tile(z_star, (1, T))  # N x T

t_train = T_star.flatten()[:, None]  # NT x 1
x_train = X_star.flatten()[:, None]  # NT x 1
y_train = Y_star.flatten()[:, None]  # NT x 1
z_train = Z_star.flatten()[:, None]  # NT x 1
t_max=t_train.max(0)
print("t_max is ", t_max)
dataB=data['data_b']
dataI=data['data_i']
print("Ne ", np.shape(t_train), "NI", np.shape(dataI), "Nb", np.shape(dataB))

tb = dataB[:, 0:1]
xb = dataB[:, 1:2]
yb = dataB[:, 2:3]
zb = dataB[:, 3:4]

# data of initial conditions
ti = dataI[:, 0:1]
xi = dataI[:, 1:2]
yi = dataI[:, 2:3]
zi = dataI[:, 3:4]

# Number of data points for the initial
Xb = np.concatenate([tb, xb, yb, zb], 1)
Xi = np.concatenate([ti, xi, yi, zi], 1)
X = np.concatenate([Xb, Xi], 0)
Xmin=X.min(0)
Xmax=X.max(0)


# Training
hvd.init()
#
totalSize=np.shape(t_train)[0]
chunkSize=totalSize // hvd.size()

startId=hvd.rank()*chunkSize
endId=startId + chunkSize

totalBSize=np.shape(dataB)[0]
totalISize=np.shape(dataI)[0]
chunkBSize=totalBSize // hvd.size()
chunkISize=totalISize // hvd.size()
startBId=hvd.rank()*chunkBSize
endBId=startBId + chunkBSize

startIId=hvd.rank()*chunkISize
endIId=startIId + chunkISize

x_loc=x_train[startId:endId, :]
y_loc=y_train[startId:endId, :]
z_loc=z_train[startId:endId, :]
t_loc=t_train[startId:endId :]
dataB_loc=dataB[startBId:endBId, :]
dataI_loc=dataI[startIId:endIId, :]


print("Local dataB id is ", np.shape(dataB_loc), "Local DATAI ", np.shape(dataI_loc), "rank is ", hvd.rank(), "Boundary Points", np.shape(x_loc))

model = ChannelFlow(t_loc, x_loc, y_loc, z_loc,
                    dataB_loc,
                    dataI_loc,
                    layers, Xmin, Xmax,
                    P_x=0.0025, niu=5e-5)


print("End loading")

start_time=time.perf_counter()
start_clock=time.process_time()

#_ = model.train(num_epochs=500, batch_number=batch_num, learning_rate=hvd.size()*1e-3)
# Warm-Up for learning rate

error = model.train(num_epochs=500, batch_number=batch_num, learning_rate=1e-3)
if hvd.rank()==0:
    model.saver.save(model.sess, './model/model.cpkt')
    scipy.io.savemat('Loss.mat', {'Loss': model.lossCol})

error = model.train(num_epochs=500, batch_number=batch_num, learning_rate=1e-4)
if hvd.rank()==0:
    model.saver.save(model.sess, './model/model.cpkt')
    scipy.io.savemat('Loss.mat', {'Loss': model.lossCol})

error = model.train(num_epochs=500, batch_number=batch_num, learning_rate=1e-5)
if hvd.rank()==0:
    model.saver.save(model.sess, './model/model.cpkt')
    scipy.io.savemat('Loss.mat', {'Loss': model.lossCol})

while error > tor:
    error = model.train(num_epochs=200, batch_number=batch_num, learning_rate=1e-5)
    if hvd.rank() == 0:
        model.saver.save(model.sess, './model/model.cpkt')
        scipy.io.savemat('Loss.mat', {'Loss': model.lossCol})

_ = model.train(num_epochs=10, batch_number=batch_num, learning_rate=1e-3)

if hvd.rank()==0:
    model.saver.save(model.sess, './model/model.cpkt')
    scipy.io.savemat('Loss.mat', {'Loss': model.lossCol})
_ = model.train(num_epochs=500, batch_number=batch_num, learning_rate=hvd.size()*1e-3)

if hvd.rank()==0:
    model.saver.save(model.sess, './model/model.cpkt')
    scipy.io.savemat('Loss.mat', {'Loss': model.lossCol})
_ = model.train(num_epochs=500, batch_number=batch_num, learning_rate=hvd.size()*1e-4)

if hvd.rank()==0:
    model.saver.save(model.sess, './model/model.cpkt')
    scipy.io.savemat('Loss.mat', {'Loss': model.lossCol})

error = model.train(num_epochs=500, batch_number=batch_num, learning_rate=hvd.size()*1e-5)
if hvd.rank()==0:
    model.saver.save(model.sess, './model/model.cpkt')
    scipy.io.savemat('Loss.mat', {'Loss': model.lossCol})

while error > tor:
     error = model.train(num_epochs=200, batch_number=batch_num, learning_rate=hvd.size()*1e-5)
     model.saver.save(model.sess, './model/model.cpkt')
     scipy.io.savemat('Loss.mat', {'Loss': model.lossCol})


stop_clock=time.process_time()
stop_time=time.perf_counter()

if hvd.rank()==0:
    print("Process time for training is ", stop_clock - start_clock)
    print("Performance time for training is ", stop_time - start_time)


