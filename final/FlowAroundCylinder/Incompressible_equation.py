"""
Author: Zongren Zou
    PINNs (physical-informed neural network) for flow around a cylinder simulation
        1. Velocity field: [u, v]
        u_x + v_y = 0, input [x, y], output [u, v]
        OR
        2. Potential function: phi
        phi_xx + phi_yy = 0, input [x, y], output phi
        Notice that it is a Laplace's equation and velocity field can be derived as [u, v] = [phi_x, phi_y]

        Boundary condition:
        u = U, v = 0 when x^2 + y^2 very large
        &
        no velocity vertical to cylinder surface when x^2 + y^2 = R^2
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt


class Model(tf.keras.Model):
    def __init__(self, r, u, u_v_b, learning_rate=1e-3):
        super(Model, self).__init__()
        # boundary
        self.R = r
        self.U = u  # velocity when [x, y] is far away from the cylinder
        self.u_b = u_v_b[:, 0]
        self.v_b = u_v_b[:, 1]
        self.lr = learning_rate

        # optimizer
        self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr)

        # network
        self.net = Sequential()
        self.net.add(Dense(32, input_shape=(2,), activation='relu'))
        for i in range(3):
            self.net.add(Dense(32, activation='relu'))
        self.net.add(Dense(2))

    def call(self, x):
        return self.net(x)

    def pde(self, tape, x):
        tape.watch(x)
        out = self.net(x)
        u, v = out[:, 0], out[:, 1]
        u_x = tape.gradient(u, x)[:, 0]
        v_y = tape.gradient(v, x)[:, 1]
        return u, v, u_x, v_y

    def loss_function(self, tape, data_train, data_b, data_c):
        # pde loss
        u, v, u_x, v_y = self.pde(tape, data_train)
        loss_pde = tf.reduce_mean((u_x + v_y) ** 2)
        # boundary loss
        u, v, u_x, v_y = self.pde(tape, data_b)
        loss_b = tf.reduce_mean((u - self.u_b) ** 2 + (v - self.v_b) ** 2)
        # loss on cylinder
        u, v, u_x, v_y = self.pde(tape, data_c)
        cos_theta = data_c[:, 0] / self.R
        sin_theta = data_c[:, 1] / self.R
        loss_c = tf.reduce_mean((u * cos_theta + v * sin_theta) ** 2)

        return loss_pde + loss_b + loss_c

    def train(self, data_train, data_b, data_c):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_function(tape, data_train, data_b, data_c)
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))
        return loss.numpy()

# def potential_flow(x, r=0.5, u=1.0):
#     """
#
#     :param x: Cartesian coordinates, numpy array
#     :param r: radius of cylinder, float
#     :param u: velocity of flow, float
#     :return: velocity field, numpy array
#     """
#     x_ = x[:, 0]
#     y_ = x[:, 1]
#     r_ = np.sqrt(x_ ** 2 + y_ ** 2)
#     sin_theta = y_ / r_
#     cos_theta = x_ / r_
#     u_r = u * (1 - (r / r_) ** 2) * cos_theta
#     u_theta = -u * (1 + (r / r_) ** 2) * sin_theta
#     u = u_r * cos_theta + u_theta * sin_theta
#     v = u_r * sin_theta - u_theta * cos_theta
#     u = np.reshape(u, [len(u), 1])
#     v = np.reshape(v, [len(v), 1])
#     return np.concatenate([u, v], axis=1)


def get_data(x_min, x_max, y_min, y_max, num_x, num_y, num_cylinder, r, u):
    x = np.linspace(x_min, x_max, num_x)
    y = np.linspace(y_min, y_max, num_y)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    inner = np.multiply(np.multiply(xx != np.min(xx), xx != np.max(xx)),
                        np.multiply(yy != np.min(yy), yy != np.max(yy)))
    x_train = xx[inner].reshape([-1, 1])
    y_train = yy[inner].reshape([-1, 1])
    data_train = np.concatenate([x_train, y_train], axis=1)

    x_b1 = xx[xx == np.min(xx)].reshape([-1, 1])
    y_b1 = yy[xx == np.min(xx)].reshape([-1, 1])
    x_b2 = xx[xx == np.max(xx)].reshape([-1, 1])
    y_b2 = yy[xx == np.max(xx)].reshape([-1, 1])
    x_b3 = xx[yy == np.min(yy)].reshape([-1, 1])
    y_b3 = yy[yy == np.min(yy)].reshape([-1, 1])
    x_b4 = xx[yy == np.max(yy)].reshape([-1, 1])
    y_b4 = yy[yy == np.max(yy)].reshape([-1, 1])
    x_b = np.concatenate([x_b1, x_b2, x_b3, x_b4], axis=0)
    y_b = np.concatenate([y_b1, y_b2, y_b3, y_b4], axis=0)
    u_b = u * np.ones_like(x_b)
    v_b = 0 * np.ones_like(x_b)
    data_b = np.concatenate([x_b, y_b, u_b, v_b], axis=1)

    theta = np.linspace(0, 2*np.pi, num_cylinder).reshape([-1, 1])
    x_c = r * np.cos(theta)
    y_c = r * np.sin(theta)
    data_c = np.concatenate([x_c, y_c], axis=1)
    return data_train, data_b, data_c


def main():
    # parameters
    num_epoch = 10000
    r = 1.0
    u = 0.5
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    num_x, num_y, num_cylinder = 60, 60, 50

    # load data
    data_train, data_b, data_c = get_data(x_min, x_max, y_min, y_max,
                                          num_x, num_y,
                                          num_cylinder,
                                          r, u)
    data_train_tensor = tf.Variable(data_train, dtype=tf.float32)
    data_b_tensor = tf.Variable(data_b[:, 0:2], dtype=tf.float32)
    u_v_b = tf.Variable(data_b[:, 2:4], dtype=tf.float32)
    data_c_tensor = tf.Variable(data_c, dtype=tf.float32)

    # build model
    model = Model(r, u, u_v_b, 1e-3)

    # training
    for i in range(num_epoch):
        current_loss = model.train(data_train_tensor, data_b_tensor, data_c_tensor)
        if (i + 1) % 100 == 0:
            print('Epoch ', i + 1, ' complete; current loss: ', current_loss)

    x = np.linspace(x_min, x_max, 10)
    y = np.linspace(y_min, y_max, 10)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten().reshape([-1, 1])
    yy = yy.flatten().reshape([-1, 1])

    data_test = np.concatenate([xx, yy], axis=1)
    data_test = tf.Variable(data_test)
    out = model.net(data_test)
    u, v = out[:, 0], out[:, 1]
    u = u.numpy()
    v = v.numpy()

    xx, yy = np.meshgrid(x, y)
    mask = xx ** 2 + yy ** 2 <= r ** 2
    u = u.reshape([10, 10])
    v = v.reshape([10, 10])
    u[mask] = np.nan
    v[mask] = np.nan

    plt.quiver(xx, yy, u, v)
    plt.grid()
    plt.savefig('velocity_field.png')
    plt.show()

    theta = np.linspace(0, 2 * np.pi, 31)
    x = r * np.cos(theta).reshape([-1, 1])
    y = r * np.sin(theta).reshape([-1, 1])
    c = np.concatenate([x, y], axis=1)
    c = tf.Variable(c, dtype=tf.float32)
    out = model.net(c)
    u = out[:, 0].numpy()
    v = out[:, 1].numpy()

    plt.quiver(x, y, u, v)
    plt.grid()
    plt.axis('equal')
    plt.savefig('velocity_around_cylinder.png')
    plt.show()


if __name__ == '__main__':
    main()


















