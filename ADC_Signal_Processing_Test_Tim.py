#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from scipy.signal import firwin
from scipy.signal import freqz
from scipy.signal import lfilter
from pylab import *
import pylab as plt
import numpy as np
import spidev
from time import sleep
import RPi.GPIO as GPIO

# uses the "spidev" package ('sudo pip3 install spidev')
# check dtparam=spi=on --> in /boot/config.txt or (set via 'sudo raspi-config')

# Set threshold for IED detection marker in ADC samples
detection_threshold = 0.01# type: float # volts

# Instatiate Rasp Pi GPIO Pins for test purposes
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(18, GPIO.OUT)

# define the sampling rate for the analog to digital converter
sampling_rate = 100000.0  # 100kHz
num_samples = 100
kHz = 1000.0 # kHz
# time for creating sin wave
time = arange(0,1,1.0/num_samples)

# Create frequencies to use for Signal Processing
f1 = 15000 # 15kHz
f2 = 30000 # 30kHz

# Create a Bandpass filter
numtaps = 64
low_cutoff = 44000  # 43.5kHz
high_cutoff = 46000  # 46.5kHz
nyq_rate = sampling_rate / 2
bpf = firwin(numtaps, [low_cutoff / nyq_rate, high_cutoff / nyq_rate], pass_zero=False)
w, h = freqz(bpf)

# Plot bandpass filter response
# plt.subplot(221)
# plot(w / (2 * pi), 20 * log10(abs(h)))
# plt.title('FIR Bandpass Filter Response')
# plt.ylabel('Magnitude (dB)')
# plt.xlabel('Frequency Coefficient')

class MCP3201(object):
    """
    Functions for reading the MCP3201 12-bit A/D converter using the SPI bus either in MSB- or LSB-mode
    """

    def __init__(self, SPI_BUS, CE_PIN):
        """
        initializes the device, takes SPI bus address (which is always 0 on newer Raspberry models)
        and sets the channel to either CE0 = 0 (GPIO pin BCM 8) or CE1 = 1 (GPIO pin BCM 7)
        """
        if SPI_BUS not in [0, 1]:
            raise ValueError('wrong SPI-bus: {0} setting (use 0 or 1)!'.format(SPI_BUS))
        if CE_PIN not in [0, 1]:
            raise ValueError('wrong CE-setting: {0} setting (use 0 for CE0 or 1 for CE1)!'.format(CE_PIN))
        self._spi = spidev.SpiDev()
        self._spi.open(SPI_BUS, CE_PIN)
        self._spi.max_speed_hz = 200000
        pass

    def readADC_MSB(self):
        """
        Reads 2 bytes (byte_0 and byte_1) and converts the output code from the MSB-mode:
        byte_0 holds two ?? bits, the null bit, and the 5 MSB bits (B11-B07),
        byte_1 holds the remaning 7 MBS bits (B06-B00) and B01 from the LSB-mode, which has to be removed.
        """
        bytes_received = self._spi.xfer2([0x00, 0x00])

        MSB_1 = bytes_received[1]
        MSB_1 = MSB_1 >> 1  # shift right 1 bit to remove B01 from the LSB mode

        MSB_0 = bytes_received[0] & 0b00011111  # mask the 2 unknown bits and the null bit
        MSB_0 = MSB_0 << 7  # shift left 7 bits (i.e. the first MSB 5 bits of 12 bits)

        return MSB_0 + MSB_1

    def readADC_LSB(self):
        """
        Reads 4 bytes (byte_0 - byte_3) and converts the output code from LSB format mode:
        byte 1 holds B00 (shared by MSB- and LSB-modes) and B01,
        byte_2 holds the next 8 LSB bits (B03-B09), and
        byte 3, holds the remaining 2 LSB bits (B10-B11).
        """
        bytes_received = self._spi.xfer2([0x00, 0x00, 0x00, 0x00])

        LSB_0 = bytes_received[1] & 0b00000011  # mask the first 6 bits from the MSB mode
        LSB_0 = bin(LSB_0)[2:].zfill(2)  # converts to binary, cuts the "0b", include leading 0s

        LSB_1 = bytes_received[2]
        LSB_1 = bin(LSB_1)[2:].zfill(8)  # see above, include leading 0s (8 digits!)

        LSB_2 = bytes_received[3]
        LSB_2 = bin(LSB_2)[2:].zfill(8)
        LSB_2 = LSB_2[0:2]  # keep the first two digits

        LSB = LSB_0 + LSB_1 + LSB_2  # concatenate the three parts to the 12-digits string
        LSB = LSB[::-1]  # invert the resulting string
        return int(LSB, base=2)

    def convert_to_voltage(self, adc_output, VREF=3.3):
        """
        Calculates analogue voltage from the digital output code (ranging from 0-4095)
        VREF could be adjusted here (standard uses the 3V3 rail from the Rpi)
        """
        return adc_output * (VREF / (2 ** 12 - 1))

   

def multiplier(signal):
    #multiply the input signal from the ADC by a 15khz sin wave to get f1 + f2 and f1 - f2
    sinwave = [np.sin(2 * np.pi * f1 * x/sampling_rate) for x in range(num_samples)]
    signal_multiplied = signal * sinwave
    return signal_multiplied


if __name__ == '__main__':
    SPI_bus = 0
    CE = 0
    MCP3201 = MCP3201(SPI_bus, CE)
    i = 0
    time_length = 1
    numpoints = 100000.0
    #array_size = int(time_length * numpoints)
    array_size = num_samples
    array = np.zeros((array_size,1))
    #array = [None]*array_size
    #time = arange(0, time_length, numpoints)

    try:
        while i != array_size:
            ADC_output_code = MCP3201.readADC_MSB()
            ADC_voltage = MCP3201.convert_to_voltage(ADC_output_code)
            # print("MCP3201 output code (MSB-mode): %d" % ADC_output_code)
            # print("MCP3201 voltage: %0.2f V" % ADC_voltage)
            #ADC_voltage = np.float32(ADC_voltage)

            array[i] = np.float32(ADC_voltage)
            # sleep(0.1)  # wait minimum of 100 ms between ADC measurements
            i = i + 1
            # print(array)

        # Read in the Least Significant Bit (not necessary for the precision needed for this)
        # ADC_output_code = MCP3201.readADC_LSB()
        # ADC_voltage = MCP3201.convert_to_voltage(ADC_output_code)
        # print("MCP3201 output code (LSB-mode): %d" % ADC_output_code)
        # print("MCP3201 voltage: %0.2f V" % ADC_voltage)
        # print()
        # sleep(0.1)

        # Multiply the noisy input signal by a 15kHz sin wave to separate frequency components (f1-f2/f1+f2
        multiplied_signal = multiplier(array)

        # # Perform an fft of the mutliplied signal (for plotting purposes otherwise commented out)
        multiplied_Signal_fft = np.fft.fft(multiplied_signal)
        # # Take absolute value of fft data or else the data is useless (complex)
        fft_multiplied_Signal = np.abs(multiplied_Signal_fft)

        # Filter the signal using the bandpass filter
        signal_filtered = lfilter(bpf, 1, multiplied_signal)

        # # Perform an fft of the filtered signal (for plotting purposes otherwise commented out)
        fft_signal = np.fft.fft(signal_filtered)
        # Take absolute value of fft data or else the data is useless (complex)
        fft_signal_filtered = np.abs(fft_signal)
        # # Plot the fft of the array data
        # plt.plot(20 * log10(abs(fft_signal_filtered)))
        # plt.title('Filtered Input Signal FFT')
        # plt.ylabel('Magnitude (dB)')
        # plt.xlabel('Frequency (Hz)')
        # plt.xlim(0,500)
        # plt.show()

        data_extracted = signal_filtered[low_cutoff/1000:high_cutoff/1000]
        threshold_avg = sum(data_extracted) / len(data_extracted)
        data_extracted.max()
        #print max(data_extracted)
        threshold_max = data_extracted.max()

        #print threshold_avg
        print threshold_max

        # Use threshold_max as a threshold for detection against set value above (detection_threshold) Should be ~ half power of 150mA but will need to tweak through testing
        if threshold_max >= detection_threshold:
            # Light up an LED on the raspberry pi if a detection is found (for testing purposes only
            # Replace this portion of code with Mavlink message transmission to pixhawk to send GCS GPS coordinates of detection
            print "LED on"
            GPIO.output(18, GPIO.HIGH)
            #time.sleep(0.5)
        else:
            print "LED off"
            GPIO.output(18, GPIO.LOW)

    except (KeyboardInterrupt):
        print('\n', "Exit on Ctrl-C: Good bye!")

    except:
        print("Other error or exception occurred!")
        raise

    finally:
        #plt.subplot(221)
        #plt.plot(time, multiplied_signal)
        #plt.title('Multiplied Input Signal')
        #plt.ylabel('Magnitude (W)')
        #plt.xlabel('Time (s)')
        #plt.subplot(222)
        #plt.plot(20 * log10(abs(multiplied_Signal_fft)))
        #plt.title('Noisy Input Signal FFT')
        #plt.ylabel('Magnitude (dB)')
        #plt.xlabel('Frequency (Hz)')
        #plt.subplot(223)
        #plt.plot(time, signal_filtered)
        #plt.title('Filtered Input Signal')
        #plt.ylabel('Magnitude (W)')
        #plt.xlabel('Time (s)')
        #plt.xlim(0,1)
        #plt.subplot(224)
        #plt.plot(20 * log10(abs(fft_signal_filtered)))
        #plt.title('Filtered Input Signal FFT')
        #plt.ylabel('Magnitude (dB)')
        #plt.xlabel('Frequency (Hz)')
        #plt.xlim(0,500)
        #plt.show()
        print()


