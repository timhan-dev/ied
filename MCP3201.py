from scipy.fftpack import rfft
from scipy.signal import firwin
from scipy.signal import freqz
from scipy. signal import lfilter
from pylab import *
import pylab as plt
import numpy as np
import spidev
from time import sleep

# uses the "spidev" package ('sudo pip3 install spidev')
# check dtparam=spi=on --> in /boot/config.txt or (set via 'sudo raspi-config')

# create a file we will use to store the data to perform the fft on
file = "ied_signal.wav"
# define the sampling rate for the analog to digital converter
sampling_rate = 100000.0 # 100kHz
amplitude = 16000
num_samples = 100000


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
        self._spi.max_speed_hz = 100000
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

# FFT function definition
#def fft_function(signal):
#    fft_signal = rfft(signal)/len(signal)
#   return fft_signal

def write_data(signal):
    # number of frames = number of samples
    nframes = num_samples

    # the following signify that the data isn't compressed to python wave functions
    comptype = "NONE"
    compname = "not compressed"
    # number of channels
    nchannels = 1
    # sampling width in bytes, wave files are usually 16 bits or 2 bytes per sample
    sampwidth = 2

    # open the file and set the parameters
    wav_file = wave.open(file, 'w')
    wav_file.setparams((nchannels, sampwidth, int(sampling_rate), nframes, comptype, compname))
    # s is the single sample being written to the file, multiplying by amplitude to convert to fixed point
    # struct takes data and packs it as binary data, 'h' means it is a 16 bit number
    # this will take our samples and write it to our file, ied_signal.wav, packed as 16 bit data.
    for s in signal:
       wav_file.writeframes(struct.pack('h', int(s*amplitude)))
    return

def fft_function():
    # frame rate same as number of samples and sampling rate of analog to digital converter (100kHz)
    frame_rate = 100000
    infile = "ied_signal.wav"
    num_samples = 100000
    wav_file = wave.open(infile, 'r')
    # wave readframes() function reads all the signal frames from a wave file
    data = wav_file.readframes(num_samples)
    wav_file.close()
    # telling the unpacker to unpack num_samples 16 bit words
    data = struct.unpack('{n}h'.format(n=num_samples), data)
    # convert the data to a numpy array.
    data = np.array(data)
    # take the fft of the data
    data_fft = np.fft.fft(data)
    # tke absolute value of fft data or else the data is useless (complex)
    frequencies = np.abs(data_fft)

    # return the frequency array element with the highest value
    print("The frequency is {} Hz".format(np.argmax(frequencies)))
    return data

if __name__ == '__main__':
    SPI_bus = 0
    CE = 0
    MCP3201 = MCP3201(SPI_bus, CE)
    i = 0
    time_length = 1
    numpoints = 1/1000
    array_size = time_length / numpoints 
    array = [None]*int(array_size)
    time = arange(0, time_length, numpoints)

    try:
        while i != array_size:
            ADC_output_code = MCP3201.readADC_MSB()
            ADC_voltage = MCP3201.convert_to_voltage(ADC_output_code)
            #print("MCP3201 output code (MSB-mode): %d" % ADC_output_code)
            #print("MCP3201 voltage: %0.2f V" % ADC_voltage)

            array[i] = ADC_voltage
            #sleep(0.1)  # wait minimum of 100 ms between ADC measurements
            i = i+1
            #print(array)
            
           # ADC_output_code = MCP3201.readADC_LSB()
           # ADC_voltage = MCP3201.convert_to_voltage(ADC_output_code)
           # print("MCP3201 output code (LSB-mode): %d" % ADC_output_code)
           # print("MCP3201 voltage: %0.2f V" % ADC_voltage)
            #print()
            
            #sleep(0.1)

        write_data(array)
        fft_function()
    except (KeyboardInterrupt):
        print('\n', "Exit on Ctrl-C: Good bye!")

    except:
        print("Other error or exception occurred!")
        raise

    finally:
        fft_signal = fft_function(array)
        plt.subplot(221)
        plt.plot(time,array)

        plt.subplot(222)
        plt.plot(20*log10(abs(fft_signal)))
        #plt.xlim(0,25)
        plt.show()
        print()
