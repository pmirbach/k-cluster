#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:32:13 2018

@author: pmirbach
"""

import numpy as np
#import edvaraux.newHelpersMulOrb as hel
from scipy.optimize import minimize





def fun1(x,a):
    return a * np.dot(x,x)

x0 = np.arange([1,1,1])

res_0 = minimize(fun1(x, a=2), x0, method='SLSQP')


