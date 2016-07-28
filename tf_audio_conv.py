import numpy as np
import tensorflow as tf
import sound

# set up variables
batch_size = 5
valid_dataset, valid_labels = sound.generate_chord_batch(batch_size)
minibatch_size = 128
num_chords = 12
minibatch = 0#tf.Placeholder

fc1_weights = tf.Variable(tf.truncated_normal([minibatch_size,num_chords]),name="fc1_weights")
fc1_biases = tf.Variable(tf.zeros([num_chords]),name="fc1_biases")

# define graph
def model(data):
	logits = tf.matmul(data,fc1_weights) + fc1_biases
	return tf.nn.softmax(logits)

def one_hot_encode(labels):
	onehot = np.zeros((labels.shape[0],num_chords))
	onehot[np.arange(labels.shape[0]),labels-1] = 1
	return onehot

#train_logits = model(minibatch)

#valid_logits = model(valid_dataset.astype(np.float32))
valid_labels = one_hot_encode(valid_labels)
print(valid_labels)

num_iterations = 10000
#for iteration in range(0,num_iterations):
#	train_minibatch, train_labels = sound.generate_chord_batch(minibatch_size)
	
