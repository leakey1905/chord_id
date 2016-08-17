import numpy as np
import sound
import cPickle

# load network
fo = open("fc1_weights",'rb')
weights1 = cPickle.load(fo)
bweights1 = np.sign(weights1)
fo.close()

fo = open("fc1_biases",'rb')
biases1 = cPickle.load(fo)
bbiases1 = np.sign(biases1)
fo.close()

fo = open("fc2_weights",'rb')
weights2 = cPickle.load(fo)
bweights2 = np.sign(weights2)
fo.close()

fo = open("fc2_biases",'rb')
biases2 = cPickle.load(fo)
bbiases2 = np.sign(biases2)
fo.close()

# ReLU activation
def relu(inputs):
	return (inputs > 0).astype(np.float32) * inputs

# test on batch of 1000 chords
batch_size=1000
chords,labels = sound.generate_chord_batch(batch_size=batch_size)
logits = np.dot(relu(np.dot(chords,weights1)+biases1),weights2) + biases2
blogits = np.dot(relu(np.dot(chords,bweights1)+bbiases1),bweights2) + bbiases2
guesses = np.argmax(logits,1)
bguesses = np.argmax(blogits,1)
acc = (guesses == labels)
bacc = (bguesses == labels)
print("Accuracy using real-valued weights/biases:")
print(100.0 * np.sum(acc.astype(np.float32)) / batch_size)
print("Accuracy using binary weights/biases:")
print(100.0 * np.sum(bacc.astype(np.float32)) / batch_size)

