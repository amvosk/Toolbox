# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 20:14:49 2021

@author: AlexVosk
"""

import math
import h5py
from tqdm import tqdm

import numpy as np

import scipy
import scipy.signal as sg
import scipy.stats as stats

import librosa
from librosa.core.spectrum import stft

import spectrum


def make_sound_mask(sound, fs, freq, noise_threshold=0.7):
    # sound required to be scaled
    #sound = sklearn.preprocessing.scale(sound)
    sound_power_smoothed = butter_filter(np.abs(sound), fs=fs, order=4, freq=freq, btype='low')
    mask = (sound_power_smoothed > sound_power_smoothed.max() * noise_threshold).astype(np.int32)
    return sound_power_smoothed, mask

def butter_filter(s, fs, order, freq, btype):
    b, a = sg.butter(order, freq / (fs / 2), btype=btype)
    sf = sg.filtfilt(b, a, s, axis=0)
    return sf

def librosa_ar_rc(y, order):

    dtype = y.dtype.type
    ar_coeffs = np.zeros(order + 1, dtype=dtype)
    ar_coeffs[0] = dtype(1)
    ar_coeffs_prev = np.zeros(order + 1, dtype=dtype)
    ar_coeffs_prev[0] = dtype(1)

    fwd_pred_error = y[1:]
    bwd_pred_error = y[:-1]

    # DEN_{M} from eqn 16 of Marple.
    den = np.dot(fwd_pred_error, fwd_pred_error) + np.dot(
        bwd_pred_error, bwd_pred_error
    )
    
    reflect_coeffs = []
    
    for i in range(order):
        if den <= 0:
            raise FloatingPointError("numerical error, input ill-conditioned?")

        reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)
        reflect_coeffs.append(reflect_coeff)
        
        ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
        for j in range(1, i + 2):
            ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]

        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

        q = dtype(1) - reflect_coeff ** 2
        den = q * den - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2

        fwd_pred_error = fwd_pred_error[1:]
        bwd_pred_error = bwd_pred_error[:-1]

    return ar_coeffs, np.array(reflect_coeffs)

def pad_same(s, frame, step, pad_type, mode='reflect'):
    if pad_type == 'left':
        return np.pad(s, (frame - step, 0), mode=mode, reflect_type='odd')
    elif pad_type == 'center':
        return np.pad(s, (math.floor((frame - step)/2), math.ceil((frame - step)/2)), mode=mode, reflect_type='odd')
    else:
        return None

class SoundFeatures:
    def __init__(self, fs, frame, step, pad='center'):
        self.fs = fs
        self.frame = frame
        self.step = step
        self.step = step
        self.pad = pad
        
        
        
    def _pad_center(self, sound):
        return np.pad(sound, (math.floor((self.frame - self.step)/2), math.ceil((self.frame - self.step)/2)), 
                      mode='reflect', reflect_type='odd')
        
    def lpc(self, sound, lpc_order=10, hanning=True):
        results = {}
        if hanning:
            hanning_window = np.hanning(self.frame)
        if self.pad == 'center':
            sound = self._pad_center(sound)
            
        n_sound_frames = (sound.shape[0] - self.frame + self.step) // self.step
        
        for coef in ['pc', 'rc', 'lsf', 'lar']:
            results[coef] = np.zeros((n_sound_frames, lpc_order))
        
        for i in range(n_sound_frames):
            sound_frame = sound[i*self.step:i*self.step + self.frame]
            if hanning:
                sound_frame = sound_frame * hanning_window
                
            ar, rc = librosa_ar_rc(sound_frame, order=lpc_order)
            pc  = - ar[1:]
            lsf = np.array(spectrum.poly2lsf(ar))
            lar = spectrum.rc2lar(rc)
            
            results['pc'][i,:] = pc
            results['rc'][i,:] = rc
            results['lsf'][i,:] = lsf
            results['lar'][i,:] = lar
        return results
        
    def ceps(self, sound, mel_fmin, mel_fmax, n_mels=40, n_mfcc=13):
        results = {}
        if self.pad == 'center':
            sound = self._pad_center(sound)
        
        melspec = librosa.feature.melspectrogram(y=sound, sr=self.fs, n_fft=self.frame, hop_length=self.step, n_mels=self.step, 
                                                 fmin=mel_fmin, fmax=mel_fmax, center=False)
        lms = np.log10(melspec + 10e-7)
        results['lms'] = lms.T
        # calculate mfcc from mel spectrogramm
        mfcc = librosa.feature.mfcc(S=lms, sr=self.fs, n_mfcc=n_mfcc)
        results['mfcc'] = mfcc.T
        return results
        
        
        

def make_sound_features(sound, fs, frame, step, pad='center', lpc_order=10, lpc_window=None, pitch_thr=0.5, soundmask_freq=10, soundmask_thr=0.7, n_mel=40, mel_fmin=0, mel_fmax=2048, n_mfcc=10):
    
#     sound = butter_filter(sound, fs, order=4, freq=0.5, btype='highpass')
#     if pad == 'left':
#         sound = np.pad(sound, (frame - step, 0), mode='reflect', reflect_type='odd')
    if pad == 'center':
        sound = np.pad(sound, (math.floor((frame - step)/2), math.ceil((frame - step)/2)), mode='reflect', reflect_type='odd')
    n_sound_frames = (sound.shape[0] - frame + step) // step

    results = {}
    results['sound'] = sound
    n_sound_frames = (sound.shape[0] - frame + step) // step

    # soundmask & featuremask
    _, soundmask = make_sound_mask(sound, fs, soundmask_freq, soundmask_thr)
    results['soundmask'] = soundmask
    featuremask = []
    for i in range (n_sound_frames):
        soundmask_frame = soundmask[i*step:i*step + frame]
        featuremask.append(np.mean(soundmask_frame) > 0.5)
    featuremask = np.array(featuremask, dtype=np.int32)
    results['featuremask'] = featuremask

    # calculate lpc
    for coef in ['pc', 'rc', 'lsf', 'lar']:
        results[coef] = np.zeros((n_sound_frames, lpc_order))
    for i in range(n_sound_frames):
        sound_frame = sound[i*step:i*step + frame]
#         print(sound_frame.shape)
        if lpc_window == 'hann':
            sound_frame = sound_frame * np.hanning(sound_frame.shape[0])
        ar, rc = librosa_ar_rc(sound_frame, order=lpc_order)
        pc  = - ar[1:]
        lsf = np.array(spectrum.poly2lsf(ar))
        lar = spectrum.rc2lar(rc)
        results['pc'][i,:] = pc
        results['rc'][i,:] = rc
        results['lsf'][i,:] = lsf
        results['lar'][i,:] = lar
        
    # calculate voiced/unvoiced and mean f0
    f0, _, voiced_probs = librosa.pyin(sound, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C4'), sr=fs, frame_length=frame, hop_length=step, center=False)
    voiced = np.zeros(voiced_probs.shape[0])
    voiced[voiced_probs < pitch_thr] = 0
    voiced[voiced_probs >= pitch_thr] = 1
    pitch = np.mean(f0[(voiced_probs > pitch_thr) * ~np.isnan(f0)])
    results['voiced'] = voiced
    if np.any(~np.isnan(f0)):
        pitch = np.mean(f0[(voiced_probs > pitch_thr) * ~np.isnan(f0)])
        results['pitch'] = pitch
    else:
        results['pitch'] = librosa.note_to_hz('C3')
    
    # calculate logmelspectrogram and mfcc
    melspec = librosa.feature.melspectrogram(y=sound, sr=fs, n_fft=frame, hop_length=step, n_mels=step, fmin=mel_fmin, fmax=mel_fmax, center=False)
    # apply log
    eps = 10e-7
    logmelspec = np.log10(melspec + eps)
    results['lms'] = logmelspec.T
    # calculate mfcc from mel spectrogramm
    mfcc = librosa.feature.mfcc(S=logmelspec, sr=fs, n_mfcc=n_mfcc)
    results['mfcc'] = mfcc.T
    return results

def make_sound(voiced, pc, frame, step, fs, pitch, intensity=20):
    
    if frame != step:
        downsample_coef = frame // step
        voiced = voiced[::downsample_coef]
        pc = pc[::downsample_coef]
    source = np.zeros(voiced.shape[0]*frame)
    print(source.shape)
    for i in range(source.shape[0]):
        if i % (fs // pitch) == 0:
            source[i] = intensity
    for i, state in enumerate(voiced):
        if state == 0:
            source[i*frame:(i+1)*frame] = np.random.normal(size=frame)
    
    result = []
    coef = np.concatenate((np.ones((pc.shape[0], 1)), -pc), axis=1)
    zi = np.zeros(coef.shape[1]-1)
    for i, pc in enumerate(coef):
        b, a = [1], pc
        sc, zi = sg.lfilter(b, a, source[i*frame:(i+1)*frame], zi=zi)
        result.append(sc)
    result = np.concatenate(result)
    return result















