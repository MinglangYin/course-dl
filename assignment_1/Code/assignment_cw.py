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
        self.input_size = 784
        self.num_classes = 10
        self.batch_size = 100
        self.learning_rate = 0.5
        # TODO: Initialize weights and biases
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros(self.num_classes)

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
        (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        temp_a = np.matmul(inputs, self.W)
        temp_b = np.zeros(temp_a.shape)
        for i in range(0, temp_b.shape[0]):
            temp_b[i,:] = self.b
        temp = temp_a+temp_b
        #temp = np.matmul(inputs, self.W)+np.transpose(np.tile(self.b, inputs.shape[0]).reshape((len(self.b), inputs.shape[0])))
        temp_exp = np.exp(temp)
        prob = np.zeros(temp_exp.shape)
        for i in range(0,prob.shape[0]):
            prob[i,:]=temp_exp[i,:]/sum(temp_exp[i,:])
        return prob
    
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
        all_loss = np.zeros(len(labels))
        for i in range(0,len(labels)):
            all_loss[i] = -np.log(probabilities[i, labels[i]])
        return np.sum(all_loss)/(len(labels))

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
        y = np.zeros((len(labels), self.num_classes))
        for i in range(0,len(labels)):
            y[i,labels[i]] = 1
        gradW = np.zeros((self.input_size, self.num_classes))
        gradB = np.zeros(self.num_classes)
        #iterate through all examples and add gradients, and average them in the end
        for i in range(0,len(labels)):
            g1 = y[i, :] - probabilities[i, :] 
            gradB = gradB + g1
            g2 = np.matmul(inputs[i, :].reshape(len(inputs[i,:]), 1), g1.reshape(1,len(g1)))
            gradW = gradW + g2
        #averaging over the batch
        gradW = gradW/len(labels)
        gradB = gradB/len(labels)
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
        correct_indicator = np.zeros(len(labels))
        for i in range(0, len(labels)):
            temp_p = probabilities[i, :]
            if np.where(temp_p == np.max(temp_p))[0][0] == labels[i]:
                correct_indicator[i] = 1
        return np.sum(correct_indicator)*1.0/(len(labels))


    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.W += self.learning_rate*gradW
        self.b += self.learning_rate*gradB




    
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
    num_examples = train_inputs.shape[0]
    nbatch = int(round(num_examples/model.batch_size))
    for i in range(0, nbatch):
        temp_id = model.batch_size*i + np.array(range(model.batch_size))
        temp_inputs = train_inputs[temp_id, :]
        temp_labels = train_labels[temp_id]
        prob = model.call(temp_inputs)
        temp_gradW, temp_gradB = model.back_propagation(temp_inputs, prob, temp_labels)
        model.gradient_descent(temp_gradW, temp_gradB)



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
    correct_indicator = np.zeros(len(test_labels))
    probabilities = model.call(test_inputs)
    for i in range(0, len(test_labels)):
        temp_p = probabilities[i, :]
        if np.where(temp_p == np.max(temp_p))[0][0] == test_labels[i]:
            correct_indicator[i] = 1
    return np.sum(correct_indicator)*1.0/(len(test_labels))
    
    

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
    train_inputs, train_labels = get_data('MNIST_data/train-images-idx3-ubyte.gz', 'MNIST_data/train-labels-idx1-ubyte.gz', 60000)
    # print(train_inputs.shape)
    # print(train_labels.shape)
    test_inputs, test_labels = get_data('MNIST_data/t10k-images-idx3-ubyte.gz', 'MNIST_data/t10k-labels-idx1-ubyte.gz', 10000)
    # print(test_inputs.shape)
    # print(test_labels.shape)
    # TODO: Create Model
    mymodel = Model()

    # TODO: Train model by calling train() ONCE on all data
    train(mymodel, train_inputs, train_labels)
    # prob = mymodel.call(train_inputs)
    # train_accuracy = mymodel.accuracy(prob, train_labels)
    # print(train_accuracy)
    # TODO: Test the accuracy by calling test() after running train()
    test_accuracy = test(mymodel, test_inputs, test_labels)
    print(test_accuracy)
    # TODO: Visualize the data by using visualize_results()
    # visual_inputs = test_inputs[range(20,30),:]
    # visual_label = test_labels[range(20,30)]
    # prob = mymodel.call(visual_inputs)
    visualize_results(visual_inputs, prob, visual_label)


    
    
if __name__ == '__main__':
    main()
