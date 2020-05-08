

import librosa
import numpy as np
import os
from sklearn.preprocessing import normalize
import random



def noise_Aug(samples, noise):
  # print("Entered noise Aug")
  if(len(noise) >= len(samples)):
    return samples
  start_index = int(random.random()*(len(samples) - len(noise)))
  samples[start_index:start_index + len(noise)] = samples[start_index:start_index + len(noise)] + noise
  return samples



def spectrogram(samples, sample_rate, overlap_ratio, window):

    stride_size = int((0.001 * sample_rate) * (window*overlap_ratio))
    window_size = int((0.001 * sample_rate) * window)
    # print("Window_size", window_size)
    # print("stride_size", stride_size)
    weighting = np.hanning(window_size)[:]
    weighting.shape = (len(weighting), 1)

    # max_freq = np.amax(samples)

    rm_entries_size = (len(samples) - window_size) % stride_size
    # print("rm",  rm_entries_size)
    padding_size = window_size - (len(samples) % stride_size)
    if(rm_entries_size > 0):
      samples = samples[:-rm_entries_size]
    # print("len", len(samples))
    nshape = (window_size, int(np.ceil(float(len(samples) - window_size)/stride_size) + 1))
    windows = np.lib.stride_tricks.as_strided(samples, shape = nshape, strides = (samples.strides[0], samples.strides[0] * stride_size)) 
    fft = np.power(np.abs(np.fft.rfft(windows * weighting,  axis = 0)), 2)
    # fft = fft/len(fft)
    fft = normalize(fft)

    # for i in range(fft.shape[0]):
    #   for j in range(fft.shape[1]):
    #     if(fft[i, j] != 0):
    #       fft[i, j] = 10*np.log10(fft[i, j])

    return fft

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
  print("num_files_noisy", num_files)
  for aud_file in training_list_2:
    # print("aud_file", aud_file)
    samples, sampling_rate = librosa.load(cur_dir + "/" + aud_file, sr = None, mono = True, offset = 0.0, duration = None)
    if(num_files > 0):
        noise_sample, sampling_rate2 = librosa.load(file_noise + "/" + noise_dir_list[int(random.random()*len(noise_dir_list))]  , sr = None, mono = True, offset = 0.0, duration = None) 
        num_files-=1
        samples = noise_Aug(samples, noise_sample) 
    specgram = spectrogram(samples, sampling_rate, 0.5, 20.0)
    np.savetxt("/content/drive/My Drive/HW2/Dataset/Spectogram/" + dir + "/" + aud_file.split(".wav")[0] + ".txt", specgram)


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
    specgram = spectrogram(samples, sampling_rate, 0.5, 20.0)
    np.savetxt("/content/drive/My Drive/HW2/Dataset/Spectogram_Val/" + dir + "/" + aud_file.split(".wav")[0] + ".txt", specgram)

# from google.colab.patches import cv2_imshow
# cv2_imshow(specgram)

# import matplotlib.pyplot as plt
# t = np.linspace(0,1,92)
# f = np.linspace(0,8000,161)
# plt.pcolormesh(t, f, specgram)
# plt.show()


#Discussed the code with Priyanshi Jain 2017358
# https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520

# https://fairyonice.github.io/implement-the-spectrogram-from-scratch-in-python.html
