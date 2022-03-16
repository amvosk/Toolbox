import warnings

import numpy as np
import pywt
from tqdm import tqdm
import scipy.signal as sg
import matplotlib.pyplot as plt

def frequencies_linspace(start, stop, nfreq):
    return np.linspace(start, stop, num=nfreq, endpoint=True)

def frequencies_logspace(start, stop, nfreq):
    return np.logspace(np.log10(start), np.log10(stop), num=nfreq, endpoint=True)

def frequencies_powerspace(start, stop, nfreq, power=2):
    return np.power(np.linspace(np.power(start, 1/power), np.power(stop, 1/power), num=nfreq, endpoint=True), power)


class AWT:
    def __init__(self, fs, temporal_resolution_1hz=None, frequency_resolution_1hz=None):
        if temporal_resolution_1hz is not None:
            if frequency_resolution_1hz is not None:
                warnings.warn('Only one type of resolutions (temporal/frequency) can be used, temporal_resolution_1hz is used')
            self.B = temporal_resolution_1hz**2 / (4*np.log(2))
        elif frequency_resolution_1hz is not None:
            self.B = (4*np.log(2)) / (np.pi*frequency_resolution_1hz)**2
        else:
            raise ValueError('No temporal/frequency resolution have been provided')

        self.fs = fs
        self.C = 1
        self.wavelet_name = f'cmor{self.B}-{self.C}'
        self.wavelet = pywt.ContinuousWavelet(self.wavelet_name)
        self.max_frequency = pywt.central_frequency(self.wavelet) * fs


    def frequency2scale(self, frequency):
        return self.max_frequency / frequency

    def scale2frequency(self, scale):
        return self.max_frequency / scale


    def temporal_full_half_hight(self, frequency):
        return np.sqrt(4*np.log(2) * self.B) / frequency

    def frequency_full_half_hight(self, frequency):
        return np.sqrt(16*np.log(2) / self.B) / (2*np.pi) * frequency


    def cwt(self, s, frequency=None, scale=np.array([1]), decimate=1):
        if frequency is not None:
            scale = self.frequency2scale(frequency)
        s = np.asarray(s)
        shape = s.shape
        
        if len(shape) == 1:
            cwtmatr, freq = pywt.cwt(s, scale, self.wavelet_name, sampling_period=1/self.fs)
            cwtmatr = cwtmatr.T
            if decimate != 1:
                cwtmatr = sg.decimate(cwtmatr, q=decimate, ftype='fir', axis=0)
        elif len(shape) > 1:
            s = s.reshape((shape[0], -1))
            
            cwtmatr = []
            for i in tqdm(range(s.shape[1])):
                c, freq = pywt.cwt(s[:, i], scale, self.wavelet_name, sampling_period=1/self.fs)
                c = c.T
                if decimate != 1:
                    c = sg.decimate(c, q=decimate, ftype='fir', axis=0)
                cwtmatr.append(c)
            cwtmatr = np.stack(cwtmatr)
            cwtmatr = np.moveaxis(cwtmatr, 0, -1)
            cwtmatr = cwtmatr.reshape((c.shape[0], scale.shape[0], *shape[1:]))
        return cwtmatr, freq


    def plot(self):
        phi, x = self.wavelet.wavefun(level=12)
        fig, ax = plt.subplots(1, 1, figsize=(15, 3))
        ax.plot(x, np.real(phi), label='real')
        ax.plot(x, np.imag(phi), label='imaginary')
        ax.legend()
        ax.grid()
        plt.show()

        
    def plot_filterbank(self, frequency):
        width = self.frequency_full_half_hight(frequency)
        sigmas = width / np.sqrt(2 * np.log(2))
        
        npoints = 1000
        a, b = np.min(frequency) - 2*np.min(width), np.max(frequency) + 2*np.max(width)
        scale = np.linspace(a, b, num=npoints, endpoint=True)
        
        def gauss_unnormalized(x, mu, s):
            return np.exp(-(x-mu)**2/(2*s**2))
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        for freq, sigma in zip(frequency, sigmas):
            ax.plot(scale, gauss_unnormalized(scale, freq, sigma), label='mu={:.2f}, sigma={:.2f}'.format(freq, sigma))
        ax.legend()
        ax.grid()
        plt.show()
        
        
    def plot_cwt(self, s, frequency=None, scale=np.array([1]), decimate=1):
        cwtmatr, _ = self.cwt(s, frequency=frequency, scale=scale, decimate=decimate)
        plt.matshow(np.abs(cwtmatr).T, aspect='auto')
        plt.show()
