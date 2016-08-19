import numpy as np
import tensorflow as tf
import sound

def one_hot_encode(labels):
	# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return labels

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# set up variables
train_batch_size = 10000
test_batch_size = 100
T = 0.1
fs = 44100
num_samples = int(T*fs)
minibatch_size = 128
num_chords = num_labels = 12
# train_dataset, train_labels = sound.generate_chord_batch(train_batch_size,T=T,fs=fs)
# train_labels = one_hot_encode(train_labels)
# valid_dataset, valid_labels = sound.generate_chord_batch(test_batch_size,T=T,fs=fs)
# valid_labels = one_hot_encode(valid_labels)

dataset, labels = sound.generate_chord_batch(train_batch_size+test_batch_size,T=T,fs=fs)
labels = one_hot_encode(labels)
train_dataset = dataset[0:train_batch_size-1]
train_labels = labels[0:train_batch_size-1]
valid_dataset = dataset[train_batch_size:train_batch_size+test_batch_size-1]
valid_labels = labels[train_batch_size:train_batch_size+test_batch_size-1]

# define DFG
graph = tf.Graph()
with graph.as_default():
	# Input data. For the training data, we use a placeholder that will be fed
	# at run time with a training minibatch.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(minibatch_size, num_samples))
	tf_train_labels = tf.placeholder(tf.float32, shape=(minibatch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)

	# Variables.
	num_hidden = 10000
	fc1_weights = tf.Variable(tf.truncated_normal([num_samples, num_hidden], dtype=tf.float32),name="fc1_weights") #seed=0 in trunc_norm
	fc1_biases = tf.Variable(tf.zeros([num_hidden]),name="fc1_biases",dtype=tf.float32)
	fc2_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], dtype=tf.float32),name="fc2_weights") #seed=0 in trunc_norm
	fc2_biases = tf.Variable(tf.zeros([num_labels]),name="fc2_biases",dtype=tf.float32)

	# Training computation.
	train_logits = tf.matmul(tf.nn.relu(tf.matmul(tf_train_dataset, fc1_weights) + fc1_biases),fc2_weights) + fc2_biases
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

	# Predictions for the training, validation, and test data.
	valid_logits = tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, fc1_weights) + fc1_biases),fc2_weights) + fc2_biases
	train_prediction = tf.nn.softmax(train_logits)
	valid_prediction = tf.nn.softmax(valid_logits)

# run DFG
num_steps = 200 + 1
with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print("Initialized")
	for step in range(num_steps):
		# Generate a minibatch.
		# batch_data, batch_labels = sound.generate_chord_batch(minibatch_size)
		# batch_labels = one_hot_encode(batch_labels)
		offset = (step * minibatch_size) % (train_labels.shape[0] - minibatch_size)
		batch_data = train_dataset[offset:(offset + minibatch_size), :]
		batch_labels = train_labels[offset:(offset + minibatch_size), :]

		# Prepare a dictionary telling the session where to feed the minibatch.
		# The key of the dictionary is the placeholder node of the graph to be fed,
		# and the value is the numpy array to feed to it.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 10 == 0):
			train_acc = accuracy(predictions, batch_labels)
			valid_acc = accuracy(valid_prediction.eval(), valid_labels)
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % train_acc)
			print("Validation accuracy: %.1f%%" % valid_acc)

	# Finally, save the weights:
	weights1 = fc1_weights.eval()
	biases1 = fc1_biases.eval()
	weights2 = fc2_weights.eval()
	biases2 = fc2_biases.eval()

	fo = open("fc1_weights",'wb')
	cPickle.dump(weights1,fo,cPickle.HIGHEST_PROTOCOL)
	fo.close()

	fo = open("fc1_biases",'wb')
	cPickle.dump(biases1,fo,cPickle.HIGHEST_PROTOCOL)
	fo.close()

	fo = open("fc2_weights",'wb')
	cPickle.dump(weights2,fo,cPickle.HIGHEST_PROTOCOL)
	fo.close()

	fo = open("fc2_biases",'wb')
	cPickle.dump(biases2,fo,cPickle.HIGHEST_PROTOCOL)
	fo.close()