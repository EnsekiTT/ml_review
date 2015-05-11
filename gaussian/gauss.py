# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
POINTS = 1000

if __name__ == '__main__':
    gauss1 = (np.random.randn(POINTS), np.random.randn(POINTS)*0.24);
    gauss2 = (np.random.randn(POINTS)*0.28, np.random.randn(POINTS));
    x1 = np.array(range(POINTS)) * 0.005
    y1 = x1 * -2
    x2 = x1
    y2 = x2 * 1
    offset_x1 = -4
    offset_y1 = 2
    cc = zip(gauss1)
    dd = zip(gauss2)
    
    cc[0] = cc[0] + x1
    cc[1] = cc[1] + y1
    cc[0] = cc[0] + offset_x1
    cc[1] = cc[1] + offset_y1

    dd[0] = dd[0] + x2
    dd[1] = dd[1] + y2

    plt.scatter(cc[0], cc[1], c=u'b')
    plt.scatter(dd[0], dd[1], c=u'r')
    plt.draw()
    plt.show()
