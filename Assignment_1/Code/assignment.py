from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with 
    batched learning. Please implement the TODOs for the entire 
    model but do not change the method and constructor arguments. 
    Make sure that your Model class works with multiple batch 
    sizes. Additionally, please exclusively use NumPy and 
    Python built-in functions for your implementation.
    """
    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 784 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 100
        self.learning_rate = 0.4
        # TODO: Initialize weights and biases
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros((1,self.num_classes))

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        probabilities = np.zeros((inputs.shape[0], 10))
        b_mat = np.tile(self.b, (inputs.shape[0], 1))
        l = np.matmul(inputs, self.W) + b_mat
        for i in range(0, inputs.shape[0]):
            total_p = sum(np.exp(l[i,:]))
            probabilities[i,:] = np.exp(l[i,:])/total_p
        return probabilities
    
    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step). 
        NOTE: This function is not actually used for gradient descent 
        in this assignment, but is a sanity check to make sure model 
        is learning.
        :param probabilities: matrix that contains the probabilities 
        per class per image of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch
        loss = np.zeros((labels.shape[0]))
        for i in range(0, labels.shape[0]):
            loss[i] = -np.log(probabilities[i,labels[i,0]])
        return np.sum(loss)/(labels.shape[0])
    
    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases 
        after one forward pass and loss calculation. The learning 
        algorithm for updating weights and biases mentioned in 
        class works for one image, but because we are looking at 
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each 
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        size_labels = labels.shape[0]
        gradW = np.zeros((self.input_size, self.num_classes))
        gradB = np.zeros((self.num_classes, 1))
        y_train = np.zeros((size_labels, self.num_classes))
        for i in range(0, size_labels):
            y_train[i, labels[i]] = 1

        for i in range (0, size_labels):
            tmp = (y_train[i,:][None,:] - probabilities[i,:])/size_labels
            gradB += np.transpose(tmp)
            gradW += np.matmul(np.transpose(inputs[i,:][None,:]), tmp)
        return gradW, gradB
        
    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        accuracy_vec = np.zeros(labels.shape[0])
        for i in range(0, labels.shape[0]):
            arr = probabilities[i,:]
            index = np.where(arr == np.amax(arr))[0][0]
            if index == labels[i]:
                accuracy_vec[i] = 1
        return np.sum(accuracy_vec)*1.0/(len(labels))

    def gradient_descent(self, gradW, gradB):
        """
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        """
        # TODO: change the weights and biases of the model to descent the gradient
        self.W += self.learning_rate*gradW
        self.b += self.learning_rate*np.transpose(gradB)
    
def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward 
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    num_x = train_inputs.shape[0]
    b_size = model.batch_size
    num_batch = int(round(num_x/b_size))
    for epoch in range(0, 1):
        for i in range(0, num_batch):
            x_train = train_inputs[i*b_size:(i+1)*b_size,:]
            y_train = train_labels[i*b_size:(i+1)*b_size,:]
            y_pred = model.call(x_train)
            loss = model.loss(y_pred, y_train)
            gradW, gradB = model.back_propagation(x_train, y_pred, y_train)
            model.gradient_descent(gradW, gradB)
        loss = model.loss(y_pred, y_train)
        print(f"epoch:{epoch}, loss:{loss}")

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment, 
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    
    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    output = model.call(test_inputs)
    accuracy = output - test_labels
    return accuracy

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model 
    for one epoch. The number of training steps should be your the number of 
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    '''

    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    inputs_file_path = './MNIST_data/train-images-idx3-ubyte.gz'
    labels_file_path = './MNIST_data/train-labels-idx1-ubyte.gz'
    num_examples = 60000
    train_inputs, train_labels = get_data(inputs_file_path, labels_file_path, num_examples)

    inputs_file_path = './MNIST_data/t10k-images-idx3-ubyte.gz'
    labels_file_path = './MNIST_data/t10k-labels-idx1-ubyte.gz'
    num_examples = 10000
    test_inputs, test_labels = get_data(inputs_file_path, labels_file_path, num_examples)
    
    # TODO: Create Model
    model = Model()

    # TODO: Train model by calling train() ONCE on all data
    train(model, train_inputs, train_labels)

    # TODO: Test the accuracy by calling test() after running train()
    probabilities = test(model, test_inputs, test_labels)
    accuracy = model.accuracy(probabilities, test_labels)
    print(accuracy)

    # TODO: Visualize the data by using visualize_results()
    input = test_inputs[range(0,10),:]
    label = test_labels[range(0,10)]
    y_pred = model.call(input)
    visualize_results(input, y_pred, label)

if __name__ == '__main__':
    main()
