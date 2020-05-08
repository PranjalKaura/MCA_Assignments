import librosa
import numpy as np
import os
from scipy.fftpack import dct
from numpy import sin
from numpy import pi
from numpy import abs
import random


def noise_Aug(samples, noise):
  if(len(noise) >= len(samples)):
    return samples
  start_index = int(random.random()*(len(samples) - len(noise)))
  samples[start_index:start_index + len(noise)] = samples[start_index:start_index + len(noise)] + noise
  return samples

def mfcc(samples, sample_rate, overlap_ratio, window):
    stride_size = int((0.001 * sample_rate) * (window*overlap_ratio))
    window_size = int((0.001 * sample_rate) * window)
    # print("Window_size", window_size)
    # print("stride_size", stride_size)
    weighting = np.hanning(window_size)[:]
    weighting.shape = (len(weighting), 1)

    # max_freq = np.amax(samples)
    num = 500
    rm_entries_size = (len(samples) - window_size) % stride_size
    # print("rm",  rm_entries_size)
    padding_size = window_size - (len(samples) % stride_size)
    if(rm_entries_size > 0):
      samples = samples[:-rm_entries_size]
    # print("len", len(samples))
    nshape = (window_size, int(np.ceil(float(len(samples) - window_size)/stride_size) + 1))       

    windows = np.lib.stride_tricks.as_strided(samples, shape = nshape, strides = (samples.strides[0], samples.strides[0] * stride_size))
    


    windows = windows*weighting
    fft = np.power(abs(np.fft.rfft(windows, num, axis = 0))/num, 2)
    fbank = get_filterbanks(num, 30, sample_rate)
    # print("fbank", fbank.shape)
    filter_banks = np.dot(fft.T, fbank.T)
    for i in range(0, filter_banks.shape[0]):
      for j in range(0, filter_banks.shape[1]):
        if(filter_banks[i][j] != 0):
          filter_banks[i, j] = 10 * np.log10(filter_banks[i, j])

    num_cep = 13
    mfcc = dct(filter_banks, axis=1, norm='ortho')[:, :num_cep]
    num_frames  = mfcc.shape[0]
    num_coeff = mfcc.shape[1]
    n = np.linspace(0, num_coeff - 1, num_coeff)
    cep_lifter = 22 
    lift_coeff = 1 + (cep_lifter/2) * sin((pi * n)/cep_lifter)
    mfcc = mfcc * lift_coeff
    return mfcc

def conv2Hz(mel):
    return 700 * (pow(10,(mel/float(2595))-1))

def conv2Mel(hz):
    return 2595 * np.log10(1 + (hz/float(700)))


def get_filterbanks(num_fft,num_filt,samplerate):
    highfreq = samplerate/2
    lowmel = 0
    highmel = conv2Mel(highfreq)
    hzpoints = conv2Hz(np.linspace(lowmel,highmel,num_filt+2))
    bin = np.floor((num_fft+1)*hzpoints/samplerate)

    fbank = np.zeros([num_filt,int(num_fft/2)+1])
    for j in range(0,num_filt):
        low_index = int(bin[j])
        high_index = int(bin[j+1])
        for i in range(low_index, high_index):
            fbank[j,i] = (i - low_index) / (high_index - low_index)
        low_index = int(bin[j+1])
        high_index = int(bin[j+2])
        for i in range(low_index, high_index):
            fbank[j,i] = (high_index-i) / (high_index - low_index)
    return fbank

file = "/content/drive/My Drive/HW2/Dataset/training"
training_list = os.listdir(file)
print(len(training_list))

file_noise = "/content/drive/My Drive/HW2/Dataset/_background_noise_"
noise_dir_list = os.listdir(file_noise)
noise_ratio = 0.0

for dir in training_list:
  cur_dir = file +"/" + dir
  print("Cur_dir", cur_dir)
  training_list_2 = os.listdir(cur_dir)
  num_files = noise_ratio*len(training_list_2)
  for aud_file in training_list_2:
    # print("aud_file", aud_file)
    samples, sampling_rate = librosa.load(cur_dir + "/" + aud_file, sr = None, mono = True, offset = 0.0, duration = None)
    if(num_files > 0):
      noise_sample, sampling_rate2 = librosa.load(file_noise + "/" + noise_dir_list[int(random.random()*len(noise_dir_list))]  , sr = None, mono = True, offset = 0.0, duration = None) 
      num_files-=1
      samples = noise_Aug(samples, noise_sample) 
    samples_2 = np.zeros(samples.shape)
    pre_emphasis = 0.96
    samples_2[0] = samples[0]
    for i in range(1, samples.shape[0]):
      samples_2[i] = samples[i] - pre_emphasis*samples[i - 1]
    MFCC = mfcc(samples_2, sampling_rate, 0.5, 20.0)
    np.savetxt("/content/drive/My Drive/HW2/Dataset/mfcc/" + dir + "/" + aud_file.split(".wav")[0] + ".txt", MFCC)



file = "/content/drive/My Drive/HW2/Dataset/validation"
training_list = os.listdir(file)
print(len(training_list))


for dir in training_list:
  cur_dir = file +"/" + dir
  print("Cur_dir", cur_dir)
  training_list_2 = os.listdir(cur_dir)
  for aud_file in training_list_2:
    # print("aud_file", aud_file)
    samples, sampling_rate = librosa.load(cur_dir + "/" + aud_file, sr = None, mono = True, offset = 0.0, duration = None)
    samples_2 = np.zeros(samples.shape)
    pre_emphasis = 0.96
    samples_2[0] = samples[0]
    for i in range(1, samples.shape[0]):
      samples_2[i] = samples[i] - pre_emphasis*samples[i - 1]
    MFCC = mfcc(samples_2, sampling_rate, 0.5, 20.0)
    np.savetxt("/content/drive/My Drive/HW2/Dataset/mfcc_val/" + dir + "/" + aud_file.split(".wav")[0] + ".txt", MFCC)


# sources: https://github.com/jameslyons/python_speech_features/blob/9a2d76c6336d969d51ad3aa0d129b99297dcf55e/python_speech_features/base.py#L149
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
#Discussed the code with Priyanshi Jain 2017358
