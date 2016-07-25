# tensorflow version of audio_conv.py to verify backprop method
import numpy as np
import tensorflow as tf

# vars for duration, sample rate, time
T = 0.1
fs = 44100
t = np.linspace(0,T,T*fs)





batch_size = 12000
valid_dataset = 0
minibatch_size = 128
training_minibatch = 0#tf.Placeholder

num_iterations = 10000

