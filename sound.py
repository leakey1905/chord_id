import numpy as np
from scipy import signal

def generate_chord(T=1,fs=44100,noise=True):
	"""
	Generate random major chord with amplitude 1
	for duration T with sampling rate fs

	Return chord and label (pitch class)
	0  = Ab	/ G#
	1  = A
	2  = Bb / A#
	3  = B
	4  = C
	5  = Db / C#
	6  = D
	7  = Eb / D#
	8  = E
	9  = F
	10 = Gb / F#
	11 = G
	"""
	t = np.linspace(0,T,T*fs)
	n1 = np.random.randint(1+17,88-10)
	chord_inversion = np.random.randint(1,6)
	n3 = 0	
	n5 = 0
	if chord_inversion == 1:
		n3 = n1 + 4
		n5 = n1 + 7
	if chord_inversion == 2:
		n3 = n1 + 7
		n5 = n1 + 16
	if chord_inversion == 3:
		n3 = n1 + -8
		n5 = n1 + 7
	if chord_inversion == 4:	
		n3 = n1 + -8
		n5 = n1 + -5
	if chord_inversion == 5:
		n3 = n1 + 4
		n5 = n1 + -5
	if chord_inversion == 6:
		n3 = n1 + -8
		n5 = n1 + -17
	f1 = 2**((n1-49)/12.0)*440
	f2 = 2**((n3-49)/12.0)*440
	f3 = 2**((n5-49)/12.0)*440
	
	phase1 = np.random.uniform(0, 2*np.pi, 1)
	phase2 = np.random.uniform(0, 2*np.pi, 1)
	phase3 = np.random.uniform(0, 2*np.pi, 1)
	label = (n1 % 12) + 1
	noise_var = np.random.uniform(0, 1, 1)

	sig = signal.square(f1*2*np.pi*t + phase1) + signal.square(f2*2*np.pi*t + phase2) + signal.square(f3*2*np.pi*t + phase3)
	if noise:
		sig = sig + noise_var*np.random.randn(int(T*fs))
	return sig/3, label-1

def generate_chord_batch(batch_size,T=0.1,fs=44100):
	chords = []
	labels = []
	# add rest of chords to batch
	for i in range(0,batch_size):
		# if i%100==0: print(i)
		chord,label = generate_chord(T=T,fs=fs)
		chords.append(chord)
		labels.append(label)
	chords = np.vstack(chords)
	labels = np.array(labels)
	return chords,labels

def play_chord(sig,fs=44100):
	import sounddevice as sd
	sd.play(0.1*sig,fs,blocking=True)

def plot_chord(sig):
	import matplotlib.pylab as plt
	t = np.linspace(0,1,sig.shape[0])
	plt.plot(t[:2000],sig[:2000])
	plt.show()

if __name__ == '__main__':
	# test function
	fs = 44100
	T = 2
	#sig1,label1 = generate_chord(T,fs)
	#print(label1)
	#play_chord(sig1)
	#plot_chord(sig1)
	chords,labels=generate_chord_batch(batch_size=10000)


