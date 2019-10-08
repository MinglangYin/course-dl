from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data
import gzip

def main():
    inputs_file_path = './MNIST_data/train-images-idx3-ubyte.gz'
    labels_file_path = './MNIST_data/train-labels-idx1-ubyte.gz'
    num_examples = 60000
    train_images, train_labels = get_data(inputs_file_path,
                                            labels_file_path,
                                            num_examples)
    
    pass

if __name__ == '__main__':
    main()
