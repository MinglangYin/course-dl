import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess
import dgl
import time

### problem unsolved:
# which relu? nn.ReLU or torch.relu



class Model(nn.Module):
	"""
	Model class representing your MPNN. This is (almost) exactly like your Model
	class from TF2, but it inherits from nn.Module instead of keras.Model.
	Treat it identically to how you treated your Model class from previous
	assignments.
	"""

	def __init__(self):
		"""
		Init method for Model class. Instantiate a lifting layer, an optimizer,
		some number of MPLayers (we recommend 3), and a readout layer going from
		the latent space of message passing to the number of classes.
		"""
		super(Model, self).__init__()

		# Initialize hyperparameters
		self.raw_features = 119
		self.num_classes = 2
		self.lr = 5e-4
		self.lift_features = 300

		# Initialize trainable parameters
		self.lift_layer = nn.Linear(self.raw_features, self.lift_features)
		self.MPLayer_1 = MPLayer(self.lift_features, self.lift_features)
		self.MPLayer_2 = MPLayer(self.lift_features, self.lift_features)
		self.MPLayer_3 = MPLayer(self.lift_features, self.lift_features)
		self.readout_layer = nn.Linear(self.lift_features, self.num_classes)
		self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)

	def forward(self, g):
		"""
		Responsible for computing the forward pass of your network. Analagous to
		"call" methods from previous assignments.

		1) The batched graph that you pass in hasn't had its node features lifted yet.
			Pop them, run them through your lifting layer. Don't apply an activation function. 
		2) After the node features for the graph have been lifted, run them through
				the mp layers (ReLUing the result returned from each).
		3) After ReLUing the result of your final mp layer, feed it through the
		   readout function in order to get logits.

		:param g: The DGL graph you wish to run inference on.
		:return: logits tensor of size (batch_size, 2)
		"""
		x = g.ndata.pop('h')
		x = self.lift_layer(x)
		x = torch.relu(self.MPLayer_1(g, x))
		x = torch.relu(self.MPLayer_2(g, x))
		x = torch.relu(self.MPLayer_3(g, x))
		y = self.readout(g, x)
		return y

	def readout(self, g, node_feats):
		""" 
		Responsible for reducing the dimensionality of the graph to
		num_classes, and summing the node features in order to return logits.

		Set your node features to be the output of your readout layer on node_feats,
		then use dgl.sum_nodes to return logits. 

		:param g: The batched DGL graph
		:param node_feats: The features at each node in the graph. Tensor of shape
			                   (num_atoms_in_batched_graph,
			                    size_of_node_vectors_from_prev_message_passing)
		:return: logits tensor of size (batch_size, 2)
		""" 
		g.ndata['h'] = self.readout_layer(node_feats)
		out = dgl.sum_nodes(g, 'h')
		return out

	def accuracy_function(self, logits, labels):
		"""
		Computes the accuracy across a batch of logits and labels.

		:param logits: a 2-D np array of size (batch_size, 2)
		:param labels: a 1-D np array of size (batch_size) 
									(1 for if the molecule is active against cancer, else 0).
		:return: mean accuracy over batch.
		"""
		times = 0
		for i, label in enumerate(labels):
			_binary = 1 if logits[i,1] > logits[i,0] else 0
			times += 1 if _binary == label else 0
		accuracy = times/labels.shape[0]
		return accuracy

class MPLayer(nn.Module):
	"""
	A PyTorch module designed to represent a single round of message passing.
	This should be instantiated in your Model class several times.
	"""

	def __init__(self, in_feats, out_feats):
		"""
		Init method for the MPLayer. You should make a layer that will be
		applied to all nodes as a final transformation when you've finished
		message passing that maps the features from size in_feats to out_feats (in case
		you want to change the dimensionality of your node vectors at between steps of message
		passing). You should also make another layer to be used in computing
		your messages.

		:param in_feats: The size of vectors at each node of your graph when you begin
		message passing for this round.
		:param out_feats: The size of vectors that you'd like to have at each of your
		nodes when you end message passing for this round.
		"""
		super(MPLayer, self).__init__()
		self.in_feats = in_feats
		self.out_feats = out_feats
		self.layer = nn.Linear(self.in_feats, self.out_feats)

	def forward(self, g, node_feats):
		"""
		Responsible for computing the forward pass of your network. Analagous to
		"call" methods from previous assignments.

		1) You should reassign g's ndata field to be the node features that were popped off
			in the previous layer (node_feats).
		2) Trigger message passing and aggregation on g using the send and recv functions.
		3) Pop the node features, and then feed it through a linear layer and return.

		You can assign/retrieve node data by accessing the graph's ndata field with some attribute
		that you'd like to save the features under in the graph (e.g g.ndata["h"] = node_feats)

		:param g: The batched DGL graph you wish to run inference on.
		:param node_feats: Beginning node features for your graph. should be a torch tensor (float) of shape
		(number_atoms_batched_graph, in_feats).
		:return: node_features of size (number_atoms_batched_graph, out_feats)
		"""
		g.ndata['h'] = node_feats
		g.send(g.edges(), self.message)
		g.recv(g.nodes(), self.reduce)
		out_feats = self.layer(g.ndata.pop('h'))
		return out_feats

	def message(self, edges):
		"""
		A function to be passed to g.send. This function, when called on a group of
		edges, should compute a message for each one of them. Each message will then be sent
		to the edge's "dst" node's mailbox.

		The particular rule to compute the message should be familiar. The message from
		node n1 with node feature v1 to n2 should be ReLU(f(v1)), where f is a feed-forward layer.

		:param edges: All the DGL edges in the batched DGL graph.
		:return: A map from some string (you choose) to all the messages
		computed for each edge. These messages can then be retrieved at
		each destination node's mailbox (e.g destination_node.mailbox["string_you_chose"]) 
		once DGL distributes them with the send function.
		"""		
		return {'dst': edges.src['h']}

	def reduce(self, nodes):
		"""
		A function to be passed to g.recv. This function, when called on a group of nodes,
		should aggregate (ie. sum) all the messages received in their mailboxes from message passing.
		DGL will then save these new features in each node under the attribute you set (see the return).

		The messages in each node can be accessed like:
		nodes.mailbox['string_you_chose_in_message']

		:param nodes: All the DGL nodes in the batched DGL Graph.
		:return: A Map from string to the summed messages for each node.
		The string should be the same attribute you've been using to
		access ndata this whole time. The node data at this
		attribute will be updated to the summed messages by DGL.
		"""
		return {'h': torch.sum(nodes.mailbox['dst'], dim=1)}

def build_graph(molecule):
	"""
	Constructs a DGL graph out of a molecule from the train/test data.

	:param molecule: a Molecule object (see molecule.py for more info)
	:return: A DGL Graph with the same number of nodes as atoms in the molecule, edges connecting them,
	         and node features applied.
	"""

	num_nodes = molecule.nodes.shape[0]

	# TODO: Initialize a DGL Graph
	g = dgl.DGLGraph()

	# TODO: Call the graph's add_nodes method with the number of nodes in the molecule.
	g.add_nodes(num_nodes)

	# TODO: Turn molecule's nodes into a tensor, and set it to be the data of this graph.
	nodes = np.asarray(molecule.nodes)
	g.ndata['h'] = torch.tensor(nodes, dtype=torch.float32)

	# TODO: Construct a tuple of src and dst nodes from the list of edges in molecules.
	#      e.g if the edges of the molecule looked like [(1,2), (3,4), (5,6)] return
	#      (1,3,5) and (2,4,6).
	edges = np.asarray(molecule.edges)
	src, dst = tuple(zip( *torch.tensor(edges, dtype=torch.float32) ))

	# TODO: Call the graph's add_edges method to add edges from src to dst and dst to src.
	#       Edges are directed in DGL, but undirected in a molecule, so you have
	#       to add them both ways.
	g.add_edges(src, dst)
	g.add_edges(dst, src)
	return g

def train(model, train_data):
	"""
	Trains your model given the training data.

	For each batch of molecules in train data...
		1) Make dgl graphs for each of the molecules in your batch; collect them in a list.
		2) call dgl.batch to turn your list of graphs into a batched graph.
		3) Turn the labels of each of the molecules in your batch into a 1-D tensor of size
		   batch_size  
		4) Pass this graph to the Model's forward pass. Run the resulting logits
				and the labels of the molecule batch through nn.CrossEntropyLoss.
		5) Zero the gradient of your optimizer.
		6) Do backprop on your loss.
		7) Take a step with the optimizer.

	Note that nn.CrossEntropyLoss expects LOGITS, not probabilities. It contains
	a softmax layer on its own. Your model won't train well if you pass it probabilities.

	:param model: Model class representing your MPNN.
	:param train_data: A 1-D list of molecule objects, representing all the molecules
	in the training set from get_data
	:return: nothing.
	"""
	bsz = 10

	## 1) build graph and collect them in a list
	data_list = []
	for molecule in train_data:
		data_list.append(molecule)

	## 2) batched graph
	num_mole = len(data_list)
	num_batch = round(num_mole/bsz)
	for i in range(num_batch):
		if i == num_batch-1:
			data_batch = data_list[i*bsz:]
		else:
			data_batch = data_list[i*bsz:(i+1)*bsz]

		labels = []
		for molecule in data_batch:
			labels.append(molecule.label)
		## 3) form a tensor
		labels = torch.tensor(np.asarray(labels), dtype=torch.long)
		
		## 4) pass to model.forward, and nn.CrossEntropyLoss
		graph_batch = []
		for molecule in data_batch:
			graph_batch.append(build_graph(molecule))
		graph_batch = dgl.batch(graph_batch)

		logits = model(graph_batch)
		loss = nn.CrossEntropyLoss()
		l = loss(logits, labels)
		print(f'loss: {l:.6f}')

		## 5) zero the gradient
		model.optimizer.zero_grad()

		## 6) backward propagation
		l.backward()

		## 7) optimizer step
		model.optimizer.step()

def test(model, test_data):
	"""
	Testing function for our model.

	Batch the molecules in test_data, feed them into your model as described in train.
	After you have the logits: turn them back into numpy arrays, compare the accuracy to the labels,
	and keep a running sum.

	:param model: Model class representing your MPNN.
	:param test_data: A 1-D list of molecule objects, representing all the molecules in your
	testing set from get_data.
	:return: total accuracy over the test set (between 0 and 1)
	"""

	bsz = 10
	accuracy = 0

	## 1) build graph and collect them in a list
	data_list = []
	for molecule in test_data:
		data_list.append(molecule)

	## 2) batched graph
	num_mole = len(data_list)
	num_batch = round(num_mole/bsz)
	for i in range(num_batch):
		if i == num_batch-1:
			data_batch = data_list[i*bsz:]
		else:
			data_batch = data_list[i*bsz:(i+1)*bsz]

		labels = []
		for molecule in data_batch:
			labels.append(molecule.label)
		## 3) form a tensor
		labels = torch.tensor(np.asarray(labels), dtype=torch.long)
		
		## 4) pass to model.forward, and nn.CrossEntropyLoss
		graph_batch = []
		for molecule in data_batch:
			graph_batch.append(build_graph(molecule))
		graph_batch = dgl.batch(graph_batch)
	
		logits = model(graph_batch)
		accuracy += model.accuracy_function(logits, labels)
	return accuracy/num_batch

def main():

	# Hyperparameters
	epochs = 15

	# TODO: Return the training and testing data from get_data
	preprocess.get_data('./1-balance.sdf')
	train_data, test_data = preprocess.get_data('./1-balance.sdf')

	# TODO: Instantiate model
	model = Model()

	# TODO: Train and test for up to 15 epochs.
	start_time = time.time()
	for epoch in range(epochs):
		print(f'epoch: {epoch}')
		train(model, train_data)
	end_time = time.time()
	training_time = (end_time-start_time)/60
	print(f'training time(min):{training_time:.6f}')

	accuracy = test(model, test_data)
	print(f'accuracy: {accuracy}')

if __name__ == '__main__':
	main()
