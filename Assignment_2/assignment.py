from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random

np.random.seed(1234)
tf.random.set_seed(1234)
batch_size = 100

class Model(tf.keras.Model):
	def __init__(self):
		"""
    	This model class will contain the architecture for your CNN that 
		classifies images. Do not modify the constructor, as doing so 
		will break the autograder. We have left in variables in the constructor
		for you to fill out, but you are welcome to change them if you'd like.
		"""
		super(Model, self).__init__()
		self.batch_size = 100
		self.num_classes = 2
		# TODO: Initialize all hyperparameters
		self.dropout_rate = 0.2

		# TODO: Initialize all trainable parameters
		self.wc1 = tf.Variable(tf.random.truncated_normal([5, 5, 3, 16], stddev=0.1, dtype=tf.float32))
		self.wc2 = tf.Variable(tf.random.truncated_normal([5, 5, 16, 20], stddev=0.1, dtype=tf.float32))
		self.wc3 = tf.Variable(tf.random.truncated_normal([5, 5, 20, 20], stddev=0.1, dtype=tf.float32))
		self.wd1 = tf.Variable(tf.random.truncated_normal([80, 32], stddev=0.1, dtype=tf.float32))
		self.wd2 = tf.Variable(tf.random.truncated_normal([32, 32], stddev=0.1, dtype=tf.float32))
		self.wd3 = tf.Variable(tf.random.truncated_normal([32, self.num_classes], stddev=0.1, dtype=tf.float32))

		self.bd1 = tf.Variable(tf.random.truncated_normal([1, 32], stddev=0.1, dtype=tf.float32))
		self.bd2 = tf.Variable(tf.random.truncated_normal([1, 32], stddev=0.1, dtype=tf.float32))
		self.bd3 = tf.Variable(tf.random.truncated_normal([1, 2], stddev=0.1, dtype=tf.float32))

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

	def call(self, inputs, is_testing=False):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
		:param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
		:return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
		"""
		# Remember that
		# shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
		# shape of filter = (filter_height, filter_width, in_channels, out_channels)
		# shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
		
		## Convolution layer 1
		x = tf.nn.conv2d(inputs, filters=self.wc1, strides=[2,2], padding='SAME')
		mean, var = tf.nn.moments(x, axes=[0])
		x = tf.nn.batch_normalization(x, mean, var, offset=0, scale=1, variance_epsilon=1e-8)
		x = tf.nn.relu(x)
		x = tf.nn.max_pool(x, ksize=[3, 3], strides=[2,2], padding='SAME')

		## Convolution layer 2
		x = tf.nn.conv2d(x, filters=self.wc2, strides=[1,1], padding='SAME')
		mean, var = tf.nn.moments(x, axes=[0])
		x = tf.nn.batch_normalization(x, mean, var, offset=0, scale=1, variance_epsilon=1e-8)
		x = tf.nn.relu(x)
		x = tf.nn.max_pool(x, ksize=[2, 2], strides=[2,2], padding='SAME')

		## Convolution layer 3
		x = tf.nn.conv2d(x, filters=self.wc3, strides=[2,2], padding='SAME')
		mean, var = tf.nn.moments(x, axes=[0])
		x = tf.nn.batch_normalization(x, mean, var, offset=0, scale=1, variance_epsilon=1e-8)
		x = tf.nn.relu(x)

		## Flatten layer
		x = tf.reshape(x, [self.batch_size, -1])
		
		## Dense Layer 1
		x = tf.nn.relu(tf.matmul(x, self.wd1) + self.bd1)
		x = tf.nn.dropout(x, self.dropout_rate)

		## Dense layer 2
		x = tf.nn.relu(tf.matmul(x, self.wd2) + self.bd2)
		x = tf.nn.dropout(x, self.dropout_rate)

		## Dense layer 3
		x = tf.matmul(x, self.wd3) + self.bd3
		logits = tf.nn.softmax(x)

		return logits
		
	def loss(self, logits, labels):
		"""
		Calculates the model cross-entropy loss after one forward pass.
		:param logits: during training, a matrix of shape (batch_size, self.num_classes) 
		containing the result of multiple convolution and feed forward layers
		Softmax is applied in this function.
		:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
		:return: the loss of the model as a Tensor
		"""
		return tf.nn.softmax_cross_entropy_with_logits(logits, labels)

	def accuracy(self, logits, labels):
		"""
		Calculates the model's prediction accuracy by comparing
		logits to correct labels – no need to modify this.
		:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
		containing the result of multiple convolution and feed forward layers
		:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

		NOTE: DO NOT EDIT
		
		:return: the accuracy of the model as a Tensor
		"""
		correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
	'''
	Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
	and labels - ensure that they are shuffled in the same order using tf.gather.
	To increase accuracy, you may want to use tf.image.random_flip_left_right on your
	inputs before doing the forward pass. You should batch your inputs.
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training), 
	shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training), 
	shape (num_labels, num_classes)
	:return: None
	'''

	with tf.GradientTape() as tape:
		logits = model.call(train_inputs)
		loss = tf.reduce_mean(model.loss(logits, train_labels))
		accuracy = model.accuracy(logits, train_labels)
		print(f'accuracy is: {accuracy:.6f}, loss is {loss:.6f}')

	gradients = tape.gradient(loss, model.trainable_variables)
	model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))	

def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels. You should NOT randomly 
	flip images or do any extra preprocessing.
	:param test_inputs: test data (all images to be tested), 
	shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	shape (num_labels, num_classes)
	:return: test accuracy - this can be the average accuracy across 
	all batches or the sum as long as you eventually divide it by batch_size
	"""
	num_batch = int(np.floor(test_inputs.shape[0]/batch_size))
	accuracy = 0
	for i in range(0, num_batch):
		batch_image = test_inputs[i*batch_size:(i+1)*batch_size]
		batch_label = test_labels[i*batch_size:(i+1)*batch_size]
		logit = model(batch_image, batch_label)
		batch_accuracy = model.accuracy(logit, batch_label)
		print(f'batch accuracy is {batch_accuracy:.6f}')
		accuracy += batch_accuracy
	test_accuracy = accuracy/num_batch
	return test_accuracy

def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
	"""
	Uses Matplotlib to visualize the results of our model.
	:param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
	:param probabilities: the output of model.call(), shape (10, num_classes)
	:param image_labels: the labels from get_data(), shape (10, num_classes)
	:param first_label: the name of the first class, "dog"
	:param second_label: the name of the second class, "cat"

	NOTE: DO NOT EDIT

	:return: doesn't return anything, a plot should pop-up 
	"""
	predicted_labels = np.argmax(probabilities, axis=1)
	num_images = image_inputs.shape[0]

	fig, axs = plt.subplots(ncols=num_images)
	fig.suptitle("PL = Predicted Label\nAL = Actual Label")
	for ind, ax in enumerate(axs):
			ax.imshow(image_inputs[ind], cmap="Greys")
			pl = first_label if predicted_labels[ind] == 0.0 else second_label
			al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
			ax.set(title="PL: {}\nAL: {}".format(pl, al))
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.tick_params(axis='both', which='both', length=0)
	plt.show()


def main():
	'''
	Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
	test your model for a number of epochs. We recommend that you train for
	10 epochs and at most 25 epochs. For CS2470 students, you must train within 10 epochs.
	You should receive a final accuracy on the testing examples for cat and dog of >=70%.
	:return: None
	'''
	## Parameters
	epochs = 10

	## import data
	first_class, second_class = 3, 5 #cat and dogs

	file_path = './CIFAR_data_compressed/train'
	image_train, label_train = get_data(file_path, first_class, second_class)

	file_path = './CIFAR_data_compressed/test'
	image_test, label_test = get_data(file_path, first_class, second_class)

	## instantiate model
	model = Model()

	## training
	num_flip = 10
	num_batch = int(np.ceil(image_train.shape[0]/batch_size))
	for epoch in range(0, epochs):
		print(f'Epoch:{epoch}')
		for i in range(0, num_batch):
			sample = np.random.choice(batch_size, num_flip)
			batch_image = image_train[i*batch_size:(i+1)*batch_size]
			batch_label = label_train[i*batch_size:(i+1)*batch_size]
			batch_image[sample] = tf.image.random_flip_left_right(batch_image[sample])
			train(model, batch_image, batch_label)
			
	## testing
	accuracy = test(model, image_test, label_test)
	print(f'overall accuracy is {accuracy:.6f}')

	# visualization
	vis_image = image_test[0:100]
	vis_label = label_test[0:100]
	prob = model.call(vis_image, vis_label)
	visualize_results(vis_image[0:10], prob[0:10], vis_label[0:10], 'dog', 'cat')

if __name__ == '__main__':
	main()
