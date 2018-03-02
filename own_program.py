#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:27:59 2018

@author: pmirbach
"""

import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from numpy import sqrt


U_vec = np.linspace(0,10,50)


def A44(U,t):
    u = U/4
    H = np.array([
        [U,     0,      0,      0,      0,      0],
        [0,     5*u-4*t,      sqrt(2)*u,    -sqrt(2)*u,   -sqrt(2)*u,   u],
        [0,     sqrt(2)*u,    U,              0,              U/2,            sqrt(2)*u],
        [0,     -sqrt(2)*u,   0,              U,              -U/2,           -sqrt(2)*u],
        [0,     -sqrt(2)*u,   U/2,            -U/2,           3*U/2,          -sqrt(2)*u],
        [0,     u,            sqrt(2)*u,    -sqrt(2)*u,   -sqrt(2)*u,   5*u+4*t]
        ])
    return H

def B44(U,t):
    u = U/4
    H = np.array([
            [U,     0,              0,                  0],
            [0,     3*u-4*t,      -sqrt(2)*u,    u],
            [0,     -sqrt(2)*u,   U/2,                sqrt(2)*u],
            [0,     u,            sqrt(2)*u,     3*u+4*t]
            ])
    return H

def C44(U,t):
    u = U/4
    H = np.array([
            [5*u-2*t,     -u,              -u,                  u],
            [-u,     5*u-2*t,      u,    -u],
            [-u,     u,   5*u+2*t,                -u],
            [u,     -u,            -u,     5*u+2*t]
            ])
    return H

def D44(U,t):
    u = U/4
    H = np.array([
            [3*u-2*t,     -u,              -u,                  u],
            [-u,          3*u-2*t,         -u,                  u],
            [-u,     -u,              3*u+2*t,                  u],
            [u,     u,              u,                  3*u+2*t],
            ])
    return H

def E44(U,t):
    u = U/4
    H = np.array([
        [3*u-4*t,     sqrt(6)*u,      u,      0,      0,      0],
        [sqrt(6)*u,    3*U/2,      -sqrt(6)*u,    0,   0,   0],
        [u,     -sqrt(6)*u,    3*u+4*t,              0,              0,            0],
        [0,     0,   0,              5*u-4*t,              -sqrt(2)*u,           u],
        [0,     0,   0,            -sqrt(2)*u,           3*U/2,          -sqrt(2)*u],
        [0,     0,   0,             u,   -sqrt(2)*u,   5*u+4*t]
        ])
    return H



Phi_0 = np.array([1,0,0,1,0,0])


Phi_1 = np.array([1,0,0,1,0,0])

Phi_2 = np.array([0,0,1,0,0,1]) 


E = np.zeros(U_vec.size)
E_0 = np.zeros(U_vec.size)
E_1 = np.zeros(U_vec.size)
E_2 = np.zeros(U_vec.size)

for i in np.arange(U_vec.size):
    
    H = E44(U_vec[i],1)
    w,v = LA.eig(H)
    
#    if i == 0:
#        E = np.zeros((U_vec.size,w.size))
    
#    E[i,:] = np.sort(w)
    E[i] = np.min(w)
    E_0[i] = np.min(np.dot(Phi_0, H * Phi_0 ))
    E_2[i] = np.min(np.dot(Phi_2, H * Phi_2 ))



#print(np.dot(HF_ansatz, H * HF_ansatz))











plt.plot(U_vec,E)
plt.plot(U_vec,E_0)
plt.plot(U_vec,E_2)
plt.show()




