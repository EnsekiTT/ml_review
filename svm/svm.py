# -*- coding:utf-8 -*-
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import inspect

# それがラムダ式か
def islambda(f):
	return inspect.isfunction(f) and f.__name__ == (lambda: True).__name__

class SVM():
    def __init__(self, kernel='liner', C=2, tol=0.01, eps=0.01):
        print "hello this is SVM code."
        self.kernel = kernel
        self.tol = tol
        self.C = C
        self.eps = eps

    def learn(self, target, point):
    	self.target = np.array(target)
    	self.target_len = len(self.target)
        self.point = np.matrix(point)

        self.alpha = np.zeros(self.target_len)
        self.b = 0
        self.error = -1*np.array(target, dtype=float)

        self.SMO()
        self.s = [i for i in range(len(self.target)) if self.eps < self.alpha[i]]
        self.m = [i for i in range(len(self.target)) if self.eps < self.alpha[i] < self.C - self.eps]
        self.b = 0.0
        for i in self.m:
            self.b += self.target[i]
            for j in self.s:
                self.b -= (self.alpha[j]*self.target[j]*self.setKernel(self.point[i], self.point[j]))
        self.b /= len(self.m)

    def takeStep(self, i1, i2):
    	if i1 == i2:
    		return 0
    	alph1 = self.alpha[i1]
    	alph2 = self.alpha[i2]
    	y1 = target[i1]
    	y2 = target[i2]
    	E1 = self.error[i1]
    	E2 = self.error[i2]
    	s = y1 * y2
    	if y1 != y2: 	# [Platt 1998] (13)
    		L = max([0, alph2 - alph1])
    		H = min([self.C, self.C + alph2 - alph1])
    	else:			# [Platt 1998] (14)
    		L = max([0, alph2 + alph1 - self.C])
    		H = min([self.C, alph2 + alph1])
    	if L == H:
    		return 0
    	k11 = self.setKernel(self.point[i1], self.point[i1])
    	k12 = self.setKernel(self.point[i1], self.point[i2])
    	k22 = self.setKernel(self.point[i2], self.point[i2])
    	eta = 2 * k12 - k11 - k22
    	if eta > 0:
    		a2 = alph2 + y2 * (E1-E2) / eta
    		if a2 < L:
    			a2 = L
    		elif a2 > H:
    			a2 = H
    	else:
    		v1 = self.evalExample(i1) + self.b - alph1 * y1 * k11 - alph2 * y2 * k12
    		v2 = self.evalExample(i2) + self.b - alph1 * y1 * k12 - alph2 * y2 * k22
    		gamma = alph1 + s * alph2
    		Lobj = self.getObj(alph1, L, i1, i2, v1, v2, s)
    		Hobj = self.getObj(alph1, H, i1, i2, v1, v2, s)
    		if Lobj > Hobj + self.eps:
    			a2 = L
    		elif Lobj < Hobj - self.eps:
    			a2 = H
    		else:
    			a2 = alph2

    	if a2 < self.eps:
    		a2 = 0
    	elif a2 > self.C - self.eps:
    		a2 = self.C

    	if np.abs(a2-alph2) < self.eps*(a2+alph2+self.eps):
    		return 0
    	a1 = alph1 + s * (alph2 - a2)
    	b = self.b
    	b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.b #[Platt 1998] (20)
    	b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.b #[Platt 1998] (21)
    	self.b = (b1 + b2) / 2

    	for i, a in enumerate(self.alpha):
    		if not (a == 0 or a == self.C):
    			self.error[i] += y1 * (a1 - alph1) * self.setKernel(self.point[i1], self.point[i]) + b
    			self.error[i] += y2 * (a2 - alph2) * self.setKernel(self.point[i2], self.point[i]) - self.b

    	self.error[i1] = 0
    	self.error[i2] = 0

    	self.alpha[i1] = a1
    	self.alpha[i2] = a2

    	return 1

    def getObj(self, alph1, O, i1, i2, v1, v2, s):
    	w = alph1 + O - 0.5 * alph1**2 * self.setKernel(self.point[i1], self.point[i1])
    	w += -0.5 * O**2 * self.setKernel(self.point[i2], self.point[i2])
    	w += -s * alph1 * O * self.setKernel(self.point[i1], self.point[i2])
    	w += -alph1 * self.target[i1] * v1
    	w += -O * self.target[i2] * v2
    	w += 0
    	return w

    def evalExample(self, index):
    	ret = 0
    	for i, a in enumerate(self.alpha):
    		if a != 0:
    			ret += a * self.target[i] * self.setKernel(self.point[index], self.point[i])
    	return ret + self.b

    def setKernel(self, k1, k2):
    	if islambda(self.kernel):
    		return self.kernel(k1, k2)
    	elif self.kernel == 'liner':
    		ret = np.dot(np.array(k1), np.array(k2).T)
    	else:
    		ret = np.dot(np.array(k1), np.array(k2).T)
    	return ret

    def examineExample(self, i2):
    	y2 = self.target[i2]
    	alph2 = self.alpha[i2]
    	E2 = self.error[i2]
    	r2 = E2*y2
    	if (r2 < -self.tol and alph2 < self.C) or (r2 >  self.tol and alph2 > 0):
    	  	nznc_alpha = [i for i in range(len(self.alpha)) if 0 != self.alpha[i] and self.C != self.alpha[i]]
    	  	if np.sum(nznc_alpha) > 1:
    	  		maxE = -np.inf
    	  		for i, E in enumerate(self.error):
    	  			if np.abs(E2 - E) > maxE:
    	  				maxE = np.abs(E2 - E)
    	  				i1 = i
    			
    			if self.takeStep(i1, i2):
    				return 1
    		
    		rnznc_alpha = nznc_alpha
    		np.random.shuffle(rnznc_alpha)
    		r_alpha = self.alpha
    		np.random.shuffle(r_alpha)
    		for i in rnznc_alpha:
    			i1 = int(i)
    			if self.takeStep(i1, i2):
    				return 1
    		for i in r_alpha:
    			i1 = int(i)
    			if self.takeStep(i1, i2):
    				return 1
    	return 0

    def SMO(self):
    	numChanged = 0
    	examineAll = 1
    	while (numChanged > 0 or examineAll):
    		numChanged = 0
    		if examineAll:
    			for I in range(self.target_len):
    				numChanged += self.examineExample(I)
    		else:
    			for I in range(self.target_len):
    				if (self.eps < self.alpha[I] < self.C - self.eps):
    					numChanged += self.examineExample(I)

    		if examineAll == 1:
    			examineAll = 0
    		elif numChanged == 0:
    			examineAll = 1


    def calc(self, x):
        ret = self.b
        for i in self.s:
            ret += (self.alpha[i]*self.target[i]*
                    self.setKernel(x, self.point[i]))
        return ret

    def get_alpha(self):
    	return self.alpha

    alpha = property(get_alpha)

if __name__ == "__main__":
	datas = pd.read_csv('../datasets/iris.txt', delim_whitespace=True)
	Label = []
	for i, d in enumerate(datas['Species']):
		if d == 'setosa':
			Label.append(0)
		elif d == 'versicolor':
			Label.append(1)
		elif d == 'virginica':
			Label.append(2)
	datas['Label'] = Label
	target = list(datas['Label'][0:100]*2.0-1.0)
	points = datas.as_matrix(columns=['Sepal.Length', 'Sepal.Width'])[0:100] #, 'Petal.Length', 'Petal.Width'

	svm = SVM()
#	print svm.setKernel(points[1], points[5])
	svm.learn(target, points)

	ans = [svm.calc(i) for i in points]
	print ans