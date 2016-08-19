import numpy as np
import sound
import cPickle
import matplotlib.pyplot as plt

# load network
fo = open("fc1_weights",'rb')
weights1 = cPickle.load(fo)
fo.close()

fo = open("fc1_biases",'rb')
biases1 = cPickle.load(fo)
fo.close()

fo = open("fc2_weights",'rb')
weights2 = cPickle.load(fo)
fo.close()

fo = open("fc2_biases",'rb')
biases2 = cPickle.load(fo)
fo.close()

# ReLU activation
def relu(inputs):
	return (inputs > 0).astype(np.float32) * inputs

# test on batch of 1000 chords
batch_size = 1000
chords,labels = sound.generate_chord_batch(batch_size=batch_size)
hidden1 = relu(np.dot(chords,weights1)+biases1)
logits = np.dot(hidden1,weights2) + biases2
guesses = np.argmax(logits,1)
acc = (guesses == labels)
print("Accuracy:")
print(100.0 * np.sum(acc.astype(np.float32)) / batch_size)

# histogram activations of output 0 when 0 was the maximum logit
indices = (labels == 0)
plt.hist(logits[indices,0])
plt.show()

# histogram activations of output 0 when another number was the label
indices = np.logical_not(indices)
plt.hist(logits[indices,0])
plt.show()

# average logit values for examples classified as chord i
t = np.arange(12)
for i in range(0,12):
	indices = (labels == i)
	plt.subplot(12,1,i+1)
	plt.plot(t,np.mean(logits[indices],0))
plt.show()
