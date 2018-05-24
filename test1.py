#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:32:13 2018

@author: pmirbach
"""

import numpy as np
#import edvaraux.newHelpersMulOrb as hel
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import scipy.linalg as lalg





#def fun1(x,a):
#    return a * np.dot(x,x)
#
#x0 = np.arange([1,1,1])
#
#res_0 = minimize(fun1(x, a=2), x0, method='SLSQP')


#a = [4,2,3]
#b = [[1,0,0],[0,1,0],[0,0,1]]
#
#c = np.dot(a,b)
#
#print(c)

#ansatz_list = ['exact', 'HF', 'exc', 'Psi_0', 'Psi_1', 'Psi_2', 'Psi_3', 'Psi_4', 'Psi_3_eff']
#print(len(ansatz_list))


#a = (0,1)
#b = (2,3)
#
#c = a+b
#
#print(c)
#
#print([1,2,3]+[4,5,6])
#
#print(c[1:2])




#a = [[1,2],[3,4]]
#print(a)
#b = np.array(a)
#print(b)



#a = np.arange(100).reshape(10,10)
#print(a)
#print(np.sum(a, axis=1))
#
#
#plt.imshow(a)
#plt.show()

#print(np.array([1,2,3])+4)
#
#a = np.zeros(20)
#inds = np.array([1,5,12])
#a[inds+3] = 1
#
#print(a)


a = np.array([[1,2],[3,4]])
b = np.array([[1,0],[0,4]])

print(np.exp(-a))
print(np.exp(-b))



print(lalg.expm(-a))
print(lalg.expm(-b))









