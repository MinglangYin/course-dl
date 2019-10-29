import tensorflow as tf
import numpy as np
from functools import reduce

def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """

    #  TODO: load and concatenate training data from training file.
    with open(train_file, 'r') as fp:
        train_list = fp.read().split()

    # TODO: load and concatenate testing data from testing file.
    with open(test_file, 'r') as fp:
        test_list = fp.read().split()
    
    vocab_dict = {}
    index = 0
    for word in train_list:
        if word in vocab_dict:
            pass
        else:
            vocab_dict[word] = index
            index += 1

    train_id = np.zeros(len(train_list), dtype=int)
    for i in range(len(train_id)):
        train_id[i] = vocab_dict[train_list[i]]

    test_id = np.zeros(len(test_list), dtype=int)
    for i in range(len(test_id)):
        test_id[i] = vocab_dict[test_list[i]]

    return train_id, test_id, vocab_dict
