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

from scipy.linalg import block_diag
from scipy.optimize import minimize

from itertools import product


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


def E44_exc(U,t):
    u = U/4
    H = np.array([
        [3*u-4*t,   u],
        [u,     3*u+4*t]
        ])
    return H



def A44_eff(U,Ue,t):
    u = U/4
    ue = Ue/4
    H = np.array([
        [U,     0,      0,      0,      0,      0],
        [0,     5*u-4*t,      sqrt(2)*ue,    -sqrt(2)*ue,   -sqrt(2)*ue,   ue],
        [0,     sqrt(2)*ue,    U,              0,              Ue/2,            sqrt(2)*ue],
        [0,     -sqrt(2)*ue,   0,              U,              -Ue/2,           -sqrt(2)*ue],
        [0,     -sqrt(2)*ue,   Ue/2,            -Ue/2,           3*U/2,          -sqrt(2)*ue],
        [0,     ue,            sqrt(2)*ue,    -sqrt(2)*ue,   -sqrt(2)*ue,   5*u+4*t]
        ])
    return H


def E44_eff(U, Ue, t):
    u = U/4
    ue = Ue/4
    H = np.array([
        [3*u-4*t,       sqrt(6)*ue,      ue,                  0, 0, 0],
        [sqrt(6)*ue,     3*U/2,          -sqrt(6)*ue,         0, 0, 0],
        [ue,             -sqrt(6)*ue,     3*u+4*t,            0, 0, 0],
        [0, 0, 0,       5*u-4*t,        -sqrt(2)*ue,           ue],
        [0, 0, 0,       -sqrt(2)*ue,     3*U/2,          -sqrt(2)*ue],
        [0, 0, 0,       ue,              -sqrt(2)*ue,     5*u+4*t]
        ])
    return H



Phi_0 = np.zeros(12)
Phi_1 = np.zeros(12)
Phi_2 = np.zeros(12)
Phi_3 = np.zeros(12)
Phi_4 = np.zeros(12)

# Phi_0 = |0011>
Phi_0[6] = Phi_0[9] = 1

# Phi_1 = |0022>
Phi_1[2] = 1

# Phi_2 = |0033>
Phi_2[6] = -1
Phi_2[9] = 1

#Phi_3 = |1122>
Phi_3[-1] = Phi_3[-4] = 1

#Phi_4 = |2233>
Phi_4[-1] = 1
Phi_4[-4] = -1



E = np.zeros(U_vec.size)
E_HF = np.zeros(U_vec.size)
E_0 = np.zeros(U_vec.size)
E_1 = np.zeros(U_vec.size)
E_2 = np.zeros(U_vec.size)
E_3 = np.zeros(U_vec.size)
E_4 = np.zeros(U_vec.size)

E_3_eff = np.zeros(U_vec.size)


def E_Psi_0(x):
    # a |0011> + b|0022>
    Psi = x[0] * Phi_0 + x[1] * Phi_1
    return np.min(np.dot(Psi, H * Psi ))

def E_Psi_1(x):
    # a |0011> + b|0033>
    Psi = x[0] * Phi_0 + x[1] * Phi_2
    return np.min(np.dot(Psi, H * Psi ))

def E_Psi_2(x):
    # a |0011> + b|1122>
    Psi = x[0] * Phi_0 + x[1] * Phi_3
    return np.min(np.dot(Psi, H * Psi ))

def E_Psi_3(x):
    # a |0011> + b|2233>
    Psi = x[0] * Phi_0 + x[1] * Phi_4
    return np.min(np.dot(Psi, H * Psi ))

def E_Psi_4(x):
    # a |0011> + b|0022> + c|0033>
    Psi = x[0] * Phi_0 + x[1] * Phi_1 + x[2] * Phi_2
    return np.min(np.dot(Psi, H * Psi ))



def E_Psi_3_eff(x):
    # a |0011> + b|2233>
    Psi = x[0] * Phi_0 + x[1] * Phi_4
    return np.min(np.dot(Psi, H_eff * Psi ))


def get_E(x, Phi_list, H):
    Psi = np.dot(x, Phi_list)
    return np.min(np.dot(Psi, H * Psi))



x_0 = np.ones(2)
cons = {'type': 'eq', 'fun': lambda x: np.dot(x,x) - 1}

x_fun_0 = []
x_fun_1 = []
x_fun_2 = []
x_fun_3 = []
x_fun_4 = []

eigs = np.zeros((12,U_vec.size))

E_exc = np.zeros(U_vec.size)

for i in np.arange(U_vec.size):
    
    H_exc = E44_exc(U_vec[i], t=1)
    w,v = LA.eig(H_exc)
    E_exc[i] = np.min(w)
    
    H1 = A44(U_vec[i],t=1)
    H2 = E44(U_vec[i],t=1)
    H = block_diag(H1,H2)
    
    w,v = LA.eig(H)
    
    idx = w.argsort()
    
    E[i] = w[idx[0]]
    eigs[:,i] = v[:,idx[0]]
    
    
    H1_eff = A44_eff(U_vec[i], U_vec[i]*1.5, t=1)
    H2_eff = E44_eff(U_vec[i], U_vec[i]*1.5, t=1)
    H_eff = block_diag(H1_eff,H2_eff)
    
#    if i == 0:
#        E = np.zeros((U_vec.size,w.size))
    
#    E[i,:] = np.sort(w)
    res_0 = minimize(E_Psi_0, x_0, method='SLSQP',constraints=cons)
#    res_1 = minimize(E_Psi_1, x_0, method='SLSQP',constraints=cons)
    res_2 = minimize(E_Psi_2, x_0, method='SLSQP',constraints=cons)
    res_3 = minimize(E_Psi_3, x_0, method='SLSQP',constraints=cons)
#    res_4 = minimize(E_Psi_4, (1,1,1), method='SLSQP',constraints=cons)
    
    res_test = minimize(get_E, x_0, args=([Phi_0, Phi_1], H), method='SLSQP',constraints=cons)
    
    res_3_eff = minimize(E_Psi_3_eff, x_0, method='SLSQP',constraints=cons)
    
    E_0[i] = res_0.fun
#    E_1[i] = res_1.fun
    E_2[i] = res_2.fun
    E_3[i] = res_3.fun
#    E_4[i] = res_4.fun
    
    E_3_eff[i] = res_3_eff.fun
    
    x_fun_0.append((res_0.x, res_0.fun))
#    x_fun_1.append((res_1.x, res_1.fun))
    x_fun_2.append((res_2.x, res_2.fun))
    x_fun_3.append((res_3.x, res_3.fun))
#    x_fun_4.append((res_4.x, res_4.fun))
    
    
    E_HF[i] = np.min(np.dot(Phi_0, H * Phi_0 ))
#    E_1[i] = np.min(np.dot(Phi_1, H * Phi_1 ))
#    E_2[i] = np.min(np.dot(Phi_2, H * Phi_2 ))
#    E_3[i] = np.min(np.dot(Phi_3, H * Phi_3 ))
#    E_4[i] = np.min(np.dot(Phi_4, H * Phi_4 ))



fig, ax = plt.subplots()

ax.plot(U_vec,E,label='exact',color='blue')
ax.plot(U_vec,E_HF,'--',label='|0011> HF',color='blue')

ax.plot(U_vec,E_0,ls=':',color='black',linewidth=3,label='a|0011> + b |0022>')
#ax.plot(U_vec,E_1,ls=':',color='red',linewidth=1,label='a|0011> + b |0033>')
ax.plot(U_vec,E_2,ls=':',color='green',linewidth=1,label='a|0011> + b |1122>')
ax.plot(U_vec,E_3,ls=':',color='grey',linewidth=1,label='a|0011> + b |2233>')

#ax.plot(U_vec,E_4,ls='-.',color='red',linewidth=1,label='a|0011> + b |0022> + c |0033>')

#ax.plot(U_vec,E_1,ls='-',color='black',linewidth=3,label='|0022>')
#ax.plot(U_vec,E_2,ls='-.',label='|0033>')
#ax.plot(U_vec,E_3,ls='-.',label='|1122>')
#ax.plot(U_vec,E_4,ls=':',label='|1133>')

ax.plot(U_vec, E_exc)


ax.plot(U_vec,E_3_eff,ls=':',color='red',linewidth=1,label='a|0011> + b |0033>')


ax.set(ylabel='energy E', xlabel='U/t')
ax.legend(loc='best')
fig.show()






#fig2, ax2 = plt.subplots()
##a_list = [el_x_fun_0[0][0] for el_x_fun_0 in x_fun_0]
##b_list = [el_x_fun_0[0][1] for el_x_fun_0 in x_fun_0]
#
#a_list = [el[0][0] for el in x_fun_3]
#b_list = [el[0][1] for el in x_fun_3]
#
#ax2.plot(a_list)
#ax2.plot(b_list,'--')
#fig2.show()




#fig3, ax3 = plt.subplots()
#
#markers = ["-", "--", "x"]
#colors = ["b", "g", "r", "c", "m", "y", "k"]
#ls = [a + b for a, b in product(markers, colors)]
#
#
#for i in range(eigs.shape[0]):
#    ax3.plot(U_vec, eigs[i,:]**2, ls[i], label=i)
#
#
#ax3.legend(loc='best')
#
#fig3.show()












