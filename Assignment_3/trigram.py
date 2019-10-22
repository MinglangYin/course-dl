import tensorflow as tf
import numpy as np
from preprocess import get_data

# tf.random.set_seed(123)
class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        """
        The Model class predicts the next words in a sequence,
        Feel free to initialize any variables that you find necessary in the constructor.

        vocab_size: The number of unique words in the data
        """
        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size
        self.vocab_size = vocab_size
        self.embedding_size = 40
        self.batch_size = 128

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.w1 = tf.Variable(tf.random.truncated_normal([2*self.embedding_size, 200], stddev=0.05, dtype=tf.float32))
        self.w2 = tf.Variable(tf.random.truncated_normal([200, self.vocab_size], stddev=0.05, dtype=tf.float32))
        # self.w3 = tf.Variable(tf.random.truncated_normal([2*self.embedding_size, self.vocab_size], stddev=0.05, dtype=tf.float32))
        
        self.b1 = tf.Variable(tf.random.truncated_normal([1, 200], stddev=0.05, dtype=tf.float32))
        self.b2 = tf.Variable(tf.random.truncated_normal([1, self.vocab_size], stddev=0.05, dtype=tf.float32))
        # self.b3 = tf.Variable(tf.random.truncated_normal([1, self.vocab_size], stddev=0.05, dtype=tf.float32))

        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=0.05, dtype=tf.float32))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self,inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)

        :return: prbs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        x = tf.nn.embedding_lookup(self.E, inputs)
        bsz = x.shape[0]
        x = tf.reshape(x, shape=[bsz, 2*self.embedding_size])

        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        # x = tf.nn.dropout(x, self.dropout_rate)

        y = tf.matmul(x, self.w2) + self.b2
        # x = tf.nn.relu(x)
        # x = tf.nn.dropout(x, self.dropout_rate)

        # x = tf.matmul(x, self.w3) + self.b3
        # y = tf.nn.softmax(x)
        return y

    def loss_function(self,logits,labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :param prbs: a matrix of shape (batch_size, vocab_size)
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
        return loss

def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples. 
    You should take the train input and shape them into groups of two words.
    Remember to shuffle your inputs and labels - ensure that they are shuffled in the same order. 
    Also you should batch your input and labels here.
    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    epochs = 1
    batch_size = 128

    num_batch = int(np.ceil(train_input.shape[0]/batch_size))
    for _ in range(0, epochs):
        for batch in range(0, num_batch):
            if batch == num_batch-1:
                batch_input = train_input[batch*batch_size:-1]
                batch_label = train_labels[batch*batch_size:-1]
            else:
                batch_input = train_input[batch*batch_size:(batch+1)*batch_size]
                batch_label = train_labels[batch*batch_size:(batch+1)*batch_size]
            
            with tf.GradientTape() as tape:
                logits = model.call(batch_input)
                loss = model.loss_function(logits, batch_label)
                print(f'loss: {loss}')

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))	

def test(model, test_input, test_labels):
    """
    Runs through all test examples. You should take the test input and shape them into groups of two words.
    And test input should be batched here as well.
    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)

    :returns: the perplexity of the test set
    """
    # num_test = test_input.shape[0]
    logits = model.call(test_input)
    print(logits.shape)
    loss = model.loss_function(logits, test_labels)
    print(loss)
    perplexity = np.exp(loss)
    return perplexity

def generate_sentence(word1, word2, length, vocab,model):
    """
    Given initial 2 words, print out predicted sentence of target length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """
    reverse_vocab = {idx:word for word, idx in vocab.items()}
    output_string = np.zeros((1,length), dtype=np.int)
    output_string[:,:2] = vocab[word1], vocab[word2]

    for end in range(2,length):
        start = end - 2
        output_string[:, end] = np.argmax(model(output_string[:,start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]
    
    print(" ".join(text))

def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess
    train_file = './data/train.txt'
    test_file = './data/test.txt'
    train_data_raw, test_data_raw, dict_vocab = get_data(train_file, test_file)
    vocab_size = len(dict_vocab)

    # TO-DO:  Separate your train and test data into inputs and labels
    train_labels = []
    train_data = []
    num_sen = len(train_data_raw)
    for i in range(0, num_sen):
        num_word = len(train_data_raw[i])
        for j in range(0, num_word-2):
            train_data.append(train_data_raw[i][j:j+2])
            train_labels.append(train_data_raw[i][j+2])
    train_data = np.asarray(train_data, dtype=int)
    train_labels = np.asarray(train_labels, dtype=int)

    test_labels = []
    test_data = []    
    num_sen = len(test_data_raw)
    for i in range(0, num_sen):
        num_word = len(test_data_raw[i])
        for j in range(0, num_word-2):
            test_data.append(test_data_raw[i][j:j+2])
            test_labels.append(test_data_raw[i][j+2])
    test_data = np.asarray(test_data, dtype=int)
    test_labels = np.asarray(test_labels, dtype=int)

    # TODO: initialize model and tensorflow variables
    model = Model(vocab_size)

    # TODO: Set-up the training step
    train(model, train_data, train_labels)

    # TODO: Set up the testing steps
    perplexity = test(model, test_data, test_labels)

    # Print out perplexity 
    print(perplexity)

    # try generate_sentence
    generate_sentence('this', 'is', 20, dict_vocab, model)

if __name__ == '__main__':
    main()


















