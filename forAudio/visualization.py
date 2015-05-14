# -*- coding:utf-8 -*-

import wave
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

chunk = 2048*2
step = 1
threshold = 5000
after = 4000
before = 1000

sound1 = '../datasets/f_sound.wav'
sound2 = '../datasets/j_sound.wav'

s1_wav = wave.open(sound1, 'rb')
s2_wav = wave.open(sound2, 'rb')

s1_frames_len = s1_wav.getnframes()
s2_frames_len = s2_wav.getnframes()

s1_aryary = []
s2_aryary = []
s1_wav.setpos(chunk-step)
while s1_wav.tell() < s1_frames_len - chunk:
	s1_wav.setpos(s1_wav.tell()-chunk+step)
	data = s1_wav.readframes(chunk)
	arya = abs(np.fromstring(data,np.int16))
	if max(arya) < threshold:
		continue

	maxary = (0,0)
	for i, m in enumerate(arya):
		if maxary[1] < m:
			maxary = (i,m)

	if len(arya)-after > maxary[0] > before:
		tempary = arya[maxary[0]-before: maxary[0]+after]
		tempary = tempary - np.mean(tempary)
		tempary = tempary / np.std(tempary)
		s1_aryary.append(tempary)


s2_wav.setpos(chunk-step)
while s2_wav.tell() < s2_frames_len - chunk:
	s2_wav.setpos(s2_wav.tell()-chunk+step)
	data = s2_wav.readframes(chunk)
	arya = abs(np.fromstring(data,np.int16))
	if max(arya) < threshold:
		continue

	maxary = (0,0)
	for i, m in enumerate(arya):
		if maxary[1] < m:
			maxary = (i,m)

	if len(arya)-after > maxary[0] > before:
		tempary = arya[maxary[0]-before: maxary[0]+after]
		tempary = tempary - np.mean(tempary)
		tempary = tempary / np.std(tempary)
		s2_aryary.append(tempary)
 

print len(s1_aryary)
s1_sum = sum(s1_aryary)
print len(s2_aryary)
s1_fft_ary = [abs(np.fft.fft(i)) for i in s1_aryary]
s2_fft_ary = [abs(np.fft.fft(i)) for i in s2_aryary]

pca = KernelPCA(n_components=3,kernel='poly')
pca.whiten=True
result = pca.fit_transform(np.array(s1_fft_ary))
x = [i[0] for i in result]
y = [i[1] for i in result]
plt.scatter(x, y, color='r')
result = pca.fit_transform(np.array(s2_fft_ary))
x = [i[0] for i in result]
y = [i[1] for i in result]
plt.scatter(x, y, color='b')
plt.show()