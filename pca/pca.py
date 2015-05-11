# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.linalg as spla
import matplotlib.pyplot as plt
from sklearn import datasets

def pca(data, base_num = 1):
	n, d = data.shape #n:データ数 d:次元数　n>dじゃないとダメです。

	data_mean = data.mean(0)
	data_norm = data - data_mean

	cov = np.dot(data_norm.T, data_norm) / float(n)
	w, vl = spla.eig(cov)
	index = w.argsort()[-min(base_num, d) :]
	t = vl[:, index[:: -1]].T
	return t

if __name__ == "__main__":
	data = np.random.multivariate_normal([0, 0], [[1, 2], [3, 4]], 100)
	iris = datasets.load_iris()
	print iris.data[:, :4]
	data = iris.data[:, :2]
	base = pca(data)
	#data = np.dot(data,base)

	#ここから可視化
	plt.scatter(data[:, 0], data[:, 1])
	leng = (data.max()-data.min())/2
	pc_line = np.array([-leng, leng]) * (base[0][1] / base[0][0])
	plt.plot([-leng, leng], pc_line, "r")
	#plt.show()

