## Questions
# 1. some of the sentences are shorter than 20 words, what should we do? Should we concatenate between the neighbors?
# 2. It seems like we don't need to initiailize weights and biases.
# 3. why do we need this dense layer? 
# 4. why we comparing a probability with the index???? in the loss function
# 5. why we need dump states out?

import tensorflow as tf
import numpy as np
from preprocess import get_data

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size
        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 50
        self.batch_size = 128

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN 
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=0.1, dtype=tf.float32))
        self.gru_1 = tf.keras.layers.GRU(self.window_size, return_sequences=True, return_state=True)
        # self.gru_2 = tf.keras.layers.GRU(self.window_size, return_sequences=True, return_state=True)
        # self.rnn_2 = tf.keras.layers.LSTM(self.window_size, return_sequences=True, return_state=True)
        # self.dense = tf.keras.layers.Dense(self.vocab_size*self.window_size)
        self.dense = tf.keras.layers.Dense(self.vocab_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.
        
        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn
        
        -Note 1: If you use an LSTM, the final_state will be the last two outputs of calling the rnn.
        If you use a GRU, it will just be the second output.

        -Note 2: You only need to use the initial state during generation. During training and testing it can be None.
        """
        x = tf.nn.embedding_lookup(self.E, inputs)
        x = self.gru_1(x)
        # x = self.gru_2(x)
        l_out, l_s = x[0], x[1]
        logits = self.dense(l_out)
        # print(logits.shape)
        # logits = tf.reshape(logits, [l_s.shape[0], self.window_size, self.vocab_size])
        return logits, l_s

    def loss(self, logits, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))
        return loss

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.
    
    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    bsz = model.batch_size
    num_batch = round(len(train_labels)/bsz)
    for i in range(num_batch):
        batch_input = train_inputs[bsz*i:(i+1)*bsz, :]
        batch_labels = train_labels[bsz*i:(i+1)*bsz, :]
        with tf.GradientTape() as tape:
            logits, state = model.call(batch_input, None)
            loss = model.loss(logits, batch_labels)
            print(f'loss: {loss}')

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))	
    

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples
    
    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set

    Note: perplexity is exp(total_loss/number of predictions)

    """
    num_predictions = test_inputs.shape[0] # need to change
    batch_size, loss = 100, 0
    num_batch = int(np.ceil(len(test_inputs)/batch_size))
    for batch in range(0, num_batch):
        print(f'batch: {batch}')
        if batch == num_batch-1:
            batch_input = test_inputs[batch*batch_size:-1]
            batch_label = test_labels[batch*batch_size:-1]
        else:
            batch_input = test_inputs[batch*batch_size:(batch+1)*batch_size]
            batch_label = test_labels[batch*batch_size:(batch+1)*batch_size]

        logits, state = model.call(batch_input, None)
        loss += model.loss(logits, batch_label)

    perplexity = np.exp(loss/num_batch)
    return perplexity

def generate_sentence(word1, length, vocab, model):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    This is only for your own exploration. What do the sequences your RNN generates look like?
    
    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits,previous_state = model.call(next_input,previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))

def main():
    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see very, very small perplexities.
    train_file = './data/train.txt'
    test_file = './data/test.txt'
    train_data_raw, test_data_raw, dict_vocab = get_data(train_file, test_file)
    vocab_size = len(dict_vocab)

    # TO-DO:  Separate your train and test data into inputs and labels
    window_size = 20
    ## train data
    train_input = []
    train_labels = []
    for i in range(len(train_data_raw)-window_size):
        train_input.append(train_data_raw[i:i+window_size])
        train_labels.append(train_data_raw[i+window_size:i+window_size+1])
    train_input = np.asarray(train_input, dtype=int)
    train_labels = np.asarray(train_labels, dtype=int)

    test_input = []
    test_labels = []
    for i in range(len(test_data_raw)-window_size):
        test_input.append(test_data_raw[i:i+window_size])
        test_labels.append(test_data_raw[i+window_size:i+window_size+1])
    test_input = np.asarray(test_input, dtype=int)
    test_labels = np.asarray(test_labels, dtype=int)

    # train_data = [sentence[:-1] for sentence in train_data_raw]
    # train_labels = [sentence[1:] for sentence in train_data_raw]
    # test_data = [sentence[:-1] for sentence in test_data_raw]
    # test_labels = [sentence[1:] for sentence in test_data_raw]

    # TODO: initialize model and tensorflow variables
    model = Model(vocab_size)

    # TODO: Set-up the training step
    train(model, train_input, train_labels)

    # TODO: Set up the testing steps
    perplexity = test(model, test_input, test_labels)

    # Print out perplexity 
    print(f'perplexity: {perplexity}')

    # try generate_sentence
    generate_sentence('this', 20, dict_vocab, model)


if __name__ == '__main__':
    main()
