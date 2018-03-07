#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:32:13 2018

@author: pmirbach
"""

import numpy as np
#import edvaraux.newHelpersMulOrb as hel
from scipy.optimize import minimize





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

ansatz_list = ['exact', 'HF', 'exc', 'Psi_0', 'Psi_1', 'Psi_2', 'Psi_3', 'Psi_4', 'Psi_3_eff']
print(len(ansatz_list))