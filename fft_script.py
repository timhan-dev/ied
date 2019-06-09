#!/usr/bin/python
## Ridiculous 6 Co. 2019

from scipy.fftpack import rfft
from scipy.signal import firwin
from scipy.signal import freqz
from scipy.signal import lfilter
from pylab import *
import pylab as plt
import numpy as np

# Create a fake signal to process using the fft
# This fake signal will be replaced with a variable input from a microcontroller pin input
time = arange(0, 1, 1.0/100000)
# time = np.linspace(0,1,1.0/1000)
f1 = 15000 # 15khz
f2 = 25000 # 20kHz
f3 = 30000 # 30kHz
f4 = 35000 # 35kHz
f5 = 40000 # 40kHz

# Create a noisy signal
signal = np.cos(f1*2*pi*time)
signal += np.cos(f2*2*pi*time)
signal += np.sin(f3*2*pi*time)
signal += np.cos(f4*2*pi*time)
signal += np.sin(f5*2*pi*time)
noise_amp = 2.0
signal += noise_amp * randn(len(time))

W = fftfreq(signal.size, d=time[1]-time[0])

# FFT fnct def
def fft_function(signal):
    fft_signal = rfft(signal)/len(signal)
    return fft_signal

# Create a Bandpass filter
numtaps = 100
fs = 100000.0 # 100kHz
low_cutoff = 28000 # 28kHz
high_cutoff = 32000 # 32kHz
nyq_rate = fs/2.0
bpf = firwin(numtaps, [low_cutoff/nyq_rate, high_cutoff/nyq_rate], pass_zero=False)
w, h = freqz(bpf)
# Plot bandpass filter response
plot(w/(2*pi), 20*log10(abs(h)))
plt.title('FIR Bandpass Filter Response')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency Coefficient')
plt.show()

# Filter the signal using the bandpass filter
signal_filtered = lfilter(bpf, 1, signal)

# Perform an fft of the original input signal
fft_signal = fft_function(signal)
# Perform an fft of the pass-band filtered input signal
fft_signal_filtered = fft_function(signal_filtered)

plt.subplot(221)
plt.plot(time,signal)
#plt.xlim(0,1)
plt.title('Noisy Input Signal')
plt.ylabel('Magnitude (W)')
plt.xlabel('Time (s)')
plt.subplot(222)
plt.plot(20*log10(abs(fft_signal)))
plt.title('Noisy Input Signal FFT')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
#plt.xlim(0,500)
plt.subplot(223)
plt.plot(time,signal_filtered)
plt.title('Filtered Input Signal')
plt.ylabel('Magnitude (W)')
plt.xlabel('Time (s)')
#plt.xlim(0,1)
plt.subplot(224)
plt.plot(20*log10(abs(fft_signal_filtered)))
plt.title('Filtered Input Signal FFT')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
#plt.xlim(0,500)
plt.show()
