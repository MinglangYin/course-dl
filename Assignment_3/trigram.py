import tensorflow as tf
import numpy as np
from preprocess import get_data

tf.random.set_seed(123)
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
        self.batch_size = 1000

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.w1 = tf.Variable(tf.random.truncated_normal([2*self.embedding_size, 200], mean=0.0, stddev=0.1, dtype=tf.float32))
        self.w2 = tf.Variable(tf.random.truncated_normal([200, self.vocab_size], mean=0.0, stddev=0.1, dtype=tf.float32)) 
        self.b1 = tf.Variable(tf.random.truncated_normal([1, 200], mean=0.0, stddev=0.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.random.truncated_normal([1, self.vocab_size], mean=0.0, stddev=0.1, dtype=tf.float32))
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], mean=0.0, stddev=0.1, dtype=tf.float32))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self,inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)

        :return: prbs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        embed_1 = tf.nn.embedding_lookup(self.E, inputs[:,0])
        embed_2 = tf.nn.embedding_lookup(self.E, inputs[:,1])
        x = tf.concat([embed_1, embed_2], axis=1)
        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        y = tf.matmul(x, self.w2) + self.b2
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
    bsz = model.batch_size
    num_batch = round(len(train_labels)/bsz)
    for i in range(num_batch):
        batch_input = train_input[bsz*i:(i+1)*bsz, :]
        batch_labels = train_labels[bsz*i:bsz*(i+1)]
        with tf.GradientTape() as tape:
            logits = model.call(batch_input)
            loss = model.loss_function(logits, batch_labels)
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
    logits = model.call(test_input)
    loss = model.loss_function(logits, test_labels)
    print(f'test loss: {loss}')
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

    train_input = []
    train_labels = []
    for i in range(len(train_data_raw)-2):
        train_input.append(train_data_raw[i:i+2])
        train_labels.append(train_data_raw[i+2])
    train_input = np.asarray(train_input, dtype=int)
    train_labels = np.asarray(train_labels, dtype=int)

    test_input = []
    test_labels = []
    for i in range(len(test_data_raw)-2):
        test_input.append(test_data_raw[i:i+2])
        test_labels.append(test_data_raw[i+2])
    test_input = np.asarray(test_input, dtype=int)
    test_labels = np.asarray(test_labels, dtype=int)

    model = Model(vocab_size)

    train(model, train_input, train_labels)

    preplexity = test(model, test_input, test_labels)

    print(preplexity)

    # try generate_sentence
    generate_sentence('this', 'is', 20, dict_vocab, model)

if __name__ == '__main__':
    main()


















