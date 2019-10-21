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

    # TODO: load and concatenate training data from training file.
    train_list = []
    with open(train_file, 'r') as fp:
        line = fp.readline()
        line = fp.readline()
        num_line = 1
        while line:
            train_list.append(line)
            line = fp.readline()
            num_line += 1

    # TODO: load and concatenate testing data from testing file.
    test_list = []
    with open(test_file, 'r') as fp:
        line = fp.readline()
        line = fp.readline()
        num_line = 1
        while line:
            test_list.append(line)
            line = fp.readline()
            num_line += 1

    # TODO: read in and tokenize training data
    dict_vocab = {}
    index = 0
    for sentence in train_list:
        token = sentence.split(' ')
        for word in token: 
            if word in dict_vocab:
                pass
            else:
                dict_vocab[word] = index
                index += 1

    # TODO: read in and tokenize testing data
    for sentence in test_list:
        token = sentence.split(' ')
        for word in token: 
            if word in dict_vocab:
                pass
            else:
                dict_vocab[word] = index
                index += 1

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.
    train_data = []
    for sentence in train_list:
        token = sentence.split(' ')
        tmp_indices = []
        for word in token:
            tmp_indices.append(dict_vocab.get(word))
        train_data.append(tmp_indices)

    test_data = []
    for sentence in test_list:
        token = sentence.split(' ')
        tmp_indices = []
        for word in token:
            tmp_indices.append(dict_vocab.get(word))
        test_data.append(tmp_indices)

    return train_data, test_data, dict_vocab