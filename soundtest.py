import numpy as np
import sounddevice as sd
import time
from scipy import signal
import matplotlib.pylab as plt

fs = 44100
phase = np.pi
T = 1
t = np.linspace(0,T,T*fs)
f = 440
sig = 0.3*signal.square(f*2*np.pi*t + phase) + 0.3*signal.square((3./4)*f*2*np.pi*t) +0.3*signal.square((5./4)*f*2*np.pi*t) 

sd.play(0.1*sig,fs,blocking=True)

plt.plot(t,sig)
plt.show()
