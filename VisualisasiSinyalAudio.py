# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:00:44 2019

@author: Aprea
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

frequency_sampling, audio_signal = wavfile.read("E:/Skripsi/DataPenelitian/P1/bukabrowser.wav")

print('\nBentuk Sinyal:', audio_signal.shape)
print('DataType Sinyal:', audio_signal.dtype)
print('Durasi Sinyal:', round(audio_signal.shape[0] / 
float(frequency_sampling), 2), 'seconds')

audio_signal = audio_signal / np.power(2, 15)

audio_signal = audio_signal [:100]
time_axis = 1000 * np.arange(0, len(audio_signal), 1) / float(frequency_sampling)

plt.plot(time_axis, audio_signal, color='blue')
plt.xlabel('Waktu (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input Sinyal Audio')
plt.show()