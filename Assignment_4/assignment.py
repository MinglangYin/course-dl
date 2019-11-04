import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import time

def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initilized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
	# and english sentences to be used by the decoder
	# - The english sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
	# 
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP] 

	# TO-DO:  Separate your train and test data into inputs and labels

	bsz = model.batch_size
	n_batch = round(len(train_french)/bsz)
	for i in range(n_batch):
		if i == n_batch-1:
			frch_batch = train_french[i*bsz:]
			eng_batch = train_english[i*bsz:]
		else:
			frch_batch = train_french[i*bsz:(i+1)*bsz]
			eng_batch = train_english[i*bsz:(i+1)*bsz]

		encoder_input = frch_batch
		decoder_input = eng_batch[:,:-1]
		labels = eng_batch[:,1:]
		mask = labels != eng_padding_index

		print(sum(mask))

		with tf.GradientTape() as tape:
			prbs = model.call(encoder_input, decoder_input)
			loss = model.loss_function(prbs, labels, mask)
			# mean_loss = tf.reduce_mean(loss)
			print(f'loss: {loss}')
		
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))	

def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initilized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:returns: perplexity of the test set, per symbol accuracy on test set
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!
	bsz, loss, accuracy = model.batch_size, 0, 0
	n_batch = int(np.ceil(len(test_english)/bsz))
	for i in range(n_batch):
		if i == n_batch-1:
			frch_batch = test_french[i*bsz:]
			eng_batch = test_english[i*bsz:]
		else:
			frch_batch = test_french[i*bsz:(i+1)*bsz]
			eng_batch = test_english[i*bsz:(i+1)*bsz]

		encoder_input = frch_batch
		decoder_input = eng_batch[:,:-1]
		labels = eng_batch[:,1:]
		mask = labels != eng_padding_index

		prbs = model.call(encoder_input, decoder_input)
		loss += model.loss_function(prbs, labels, mask)

		accuracy = model.accuracy_function(prbs, labels, mask)
		print(f'loss: {loss}, accuracy: {accuracy}')
		
	perplexity = np.exp(loss/n_batch)
	
	return perplexity, accuracy


def main():	
	if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [RNN/TRANSFORMER]")
			exit()

	print("Running preprocessing...")
	train_english, test_english, train_french, test_french, english_vocab, french_vocab,eng_padding_index = get_data('data/fls.txt','data/els.txt','data/flt.txt','data/elt.txt')
	print("Preprocessing complete.")

	model_args = (FRENCH_WINDOW_SIZE,len(french_vocab),ENGLISH_WINDOW_SIZE, len(english_vocab))
	if sys.argv[1] == "RNN":
		model = RNN_Seq2Seq(*model_args)
	elif sys.argv[1] == "TRANSFORMER":
		model = Transformer_Seq2Seq(*model_args) 
	
	# TODO:
	# Train and Test Model for 1 epoch.
	epochs = 1
	start_time = time.time()
	for _ in range(epochs):
		train(model, train_french, train_english, eng_padding_index)
	end_time = time.time()
	print(f'training time: {end_time-start_time}')

	perplexity, accu_persym = test(model, test_french, test_english, eng_padding_index)
	print(f'perplexity: {perplexity}, accuracy_per_symbol: {accu_persym}')

if __name__ == '__main__':
   main()


