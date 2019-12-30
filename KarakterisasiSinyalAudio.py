# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:00:44 2019

@author: Aprea
"""
#import library
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#baca file audio yang disimpan. ini akan mengembalikan dua nilai: frekuensi sampling dan sinyal audio.
frequency_sampling, audio_signal = wavfile.read("E:/Skripsi/DataPenelitian/P1/mulai.wav")
#menampilkan parameter seperti frekuensi sampling sinyal audio, tipe data sinyal dan durasinya
print('\nBentuk Sinyal:', audio_signal.shape)
print('DataType Sinyal:', audio_signal.dtype)
print('Durasi Sinyal:', round(audio_signal.shape[0] / 
float(frequency_sampling), 2), 'seconds')
#menormalkan sinyal
audio_signal = audio_signal / np.power(2, 15)
#ekstraksi panjang sinyal dan setengah dari panjang sinyal
length_signal = len(audio_signal)
half_length = np.ceil((length_signal + 1) / 2.0).astype(np.int)
#menerapkan alat matematika untuk mentransformasikannya menjadi domain frekuensi(transformasi fourier)
signal_frequency = np.fft.fft(audio_signal)
#normalisasi sinyal domain frekuensi dan kuadratkan
signal_frequency = abs(signal_frequency[0:half_length]) / length_signal
signal_frequency **= 2
#ekstrak panjang dan setengah panjang dari frekuensi sinyal yang ditransformasikan
len_fts = len(signal_frequency)
#sinyal transformasi Fourier harus disesuaikan untuk kasus genap maupun ganjil
if length_signal % 2:
   signal_frequency[1:len_fts] *= 2
else:
   signal_frequency[1:len_fts-1] *= 2
   #ekstrak power dalam desibal(dB)
   signal_power = 10 * np.log10(signal_frequency)
   #Sesuaikan frekuensi dalam kHz untuk sumbu X
   x_axis = np.arange(0, half_length, 1) * (frequency_sampling / length_signal) / 1000.0
   #visualisasi karakterisasi sinyal sebagai berikut
   plt.figure()
   plt.plot(x_axis, signal_power, color='blue')
   plt.xlabel('Frequensi (kHz)')
   plt.ylabel('Daya (Power) Sinyal (dB)')
   plt.show()
   