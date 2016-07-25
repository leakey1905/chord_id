import numpy as np
import sounddevice as sd
import time
from scipy import signal

fs = 44100
T = 3
t = np.linspace(0,T,T*fs)
f = 440
sig = 0.3*signal.square(f*2*np.pi*t) + 0.3*signal.square((3./4)*f*2*np.pi*t) +0.3*signal.square((5./4)*f*2*np.pi*t) 

sd.play(0.1*sig,fs,blocking=True)
