#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:27:59 2018

@author: pmirbach
"""

import numpy as np
from numpy import linalg as LA

from H44_parts import A44, E44, E44_exc, A44_eff, E44_eff

from matplotlib import pyplot as plt
#from numpy import sqrt

from scipy.linalg import block_diag
from scipy.optimize import minimize

from itertools import product
from plot_scripts import plot_font_size


U_vec = np.linspace(0,8,200)


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





def get_H(U, t=1):
    H1 = A44(U, t)
    H2 = E44(U, t)
    return block_diag(H1,H2)

def get_H_eff(U, U_eff, t=1):
    H1 = A44_eff(U, U_eff, t)
    H2 = E44_eff(U, U_eff, t)
    return block_diag(H1,H2)


def get_E(x, Phi_list, H):
    Psi = np.dot(x, Phi_list)
    return np.min(np.dot(Psi, H * Psi))



x_0 = np.ones(2)
cons = {'type': 'eq', 'fun': lambda x: np.dot(x,x) - 1}



eigs = np.zeros((12,U_vec.size))

ansatz_list = ['exact', 'HF', 'exc', 'Psi_0', 'Psi_1', 'Psi_2', 'Psi_3', 'Psi_4', 'Psi_3_eff']
N_a = len(ansatz_list)
E = np.zeros((U_vec.size, N_a))

#E = np.zeros(U_vec.size)
#E_HF = np.zeros(U_vec.size)
#E_exc = np.zeros(U_vec.size)
#
#E_0 = np.zeros(U_vec.size)
#E_1 = np.zeros(U_vec.size)
#E_2 = np.zeros(U_vec.size)
#E_3 = np.zeros(U_vec.size)
#E_4 = np.zeros(U_vec.size)
#
#E_3_eff = np.zeros(U_vec.size)


for i in np.arange(U_vec.size):
    U = U_vec[i]
    
    # Get Hamiltonians
    H = get_H(U)    
    H_exc = E44_exc(U, t=1)
    
    U_eff = U * 1.2
    H_eff = get_H_eff(U, U_eff)
    
    
    # Exact solution
    w,v = LA.eig(H)
    idx = w.argsort()
    
    E[i,0] = w[idx[0]]
    eigs[:,i] = v[:,idx[0]]    
    
    
    # Hartree Fock
    E[i,1] = np.min(np.dot(Phi_0, H * Phi_0 ))
    
    
    # Excerpt of H with simple basisstates
    w,v = LA.eig(H_exc)
    E[i,2] = np.min(w)
    
    
    # Phi_0 = |0011>
    # Phi_1 = |0022>
    # Phi_2 = |0033>
    # Phi_4 = |2233>
    
    res_0 = minimize(get_E, [1,1], args=([Phi_0, Phi_1], H), method='SLSQP',constraints=cons)
    
#    res_1 = minimize(get_E, [1,1], args=([Phi_0, Phi_4], H), method='SLSQP',constraints=cons)
    
#    res_2 = minimize(get_E, [1,1], args=([Phi_0, Phi_3], H), method='SLSQP',constraints=cons)
    res_3 = minimize(get_E, [1,1,1], args=([Phi_0, Phi_1, Phi_3, Phi_4], H), method='SLSQP',constraints=cons)
#    res_3 = minimize(get_E, [1,1], args=([Phi_0, Phi_4], H), method='SLSQP',constraints=cons)
#    res_4 = minimize(get_E, [1,1], args=([Phi_2, Phi_4], H), method='SLSQP',constraints=cons)
    
#    res_3_eff = minimize(get_E, [1,1], args=([Phi_0, Phi_4], H_eff), method='SLSQP',constraints=cons)
    res_3_eff = minimize(get_E, [1,1,1,1], args=([Phi_0, Phi_1, Phi_3, Phi_4], H_eff), method='SLSQP',constraints=cons)


#    res_3_eff = minimize(E_Psi_3_eff, x_0, method='SLSQP',constraints=cons)
    
    E[i,3] = res_0.fun
#    E[i,4] = res_1.fun
#    E[i,5] = res_2.fun
    E[i,6] = res_3.fun
#    E[i,7] = res_4.fun
    
    E[i,8] = res_3_eff.fun



#fig, ax = plt.subplots()
#
#ax.plot(U_vec, E[:,0], label='exact', color='blue')
#ax.plot(U_vec, E[:,1], '--', label='|0011> HF', color='blue')
#
#ax.plot(U_vec, E[:,3], ls=':', color='black', linewidth=3, label='a|0011> + b |0022>')
##ax.plot(U_vec,E_1,ls=':',color='red',linewidth=1,label='a|0011> + b |0033>')
#ax.plot(U_vec, E[:,5], ls=':', color='green', linewidth=1, label='a|0011> + b |1122>')
#ax.plot(U_vec, E[:,6], ls=':', color='grey', linewidth=1,label='a|0011> + b |2233>')
#
##ax.plot(U_vec,E_4,ls='-.',color='red',linewidth=1,label='a|0011> + b |0022> + c |0033>')
#
##ax.plot(U_vec,E_1,ls='-',color='black',linewidth=3,label='|0022>')
##ax.plot(U_vec,E_2,ls='-.',label='|0033>')
##ax.plot(U_vec,E_3,ls='-.',label='|1122>')
##ax.plot(U_vec,E_4,ls=':',label='|1133>')
#
#ax.plot(U_vec, E[:,2], label='Excerpt')
#
#
#ax.plot(U_vec,E[:,8], ls=':', color='red', linewidth=1, label='a|0011> + b |0033>')
#
#
#ax.set(ylabel='energy E', xlabel='U/t')
#ax.legend(loc='best')
#fig.show()



def get_cdd(a):
    s = r'\(c_{' + str(a) + r'\uparrow}^{\dagger}c_{' + str(a) + r'\downarrow}^{\dagger}\)'
    return s

def ket():
    s = r'\left|0\right\rangle'
    return s

#print(ket(1))

a = get_cdd(0)


plot_font_size('poster')
lw = 3

plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{braket}')
#plt.verbose.level = 'debug-annoying'
#plt.rc('font', family='serif')



fig, ax = plt.subplots()

#for i in range(5):
#    ax.plot(U_vec, E[:,i], label=ansatz_list[i])


ax.plot(U_vec, E[:,0], linewidth=lw, label=r'Exact solution', color='black')
#ax.plot(U_vec, E[:,1], '--', label='|0011> HF', color='blue')
HF_theo = -4 + U_vec*3/4
ax.plot(U_vec, HF_theo, '-', linewidth=lw, label=r'Hartree Fock', color='forestgreen')
#ax.plot(U_vec, HF_theo, '-', linewidth=lw, label=r'Hartree Fock: \Psi = ' + get_cdd(0) + get_cdd(1) + ket(), color='red')
ax.plot(U_vec, E[:,2], linewidth=lw, label=r'Reduced Fock space', color='orange')

#ax.plot(U_vec, E[:,3], ls=':', color='black', linewidth=3, label='a|0011> + b |0022>')
#ax.plot(U_vec,E_1,ls=':',color='red',linewidth=1,label='a|0011> + b |0033>')
#ax.plot(U_vec, E[:,5], ls=':', color='green', linewidth=1, label='a|0011> + b |1122>')
ax.plot(U_vec, E[:,6], ls='-', color='blue', linewidth=lw, label=r'Variational state: ' + r'\left| \Psi_{1} \right \rangle')

#ax.plot(U_vec,E_4,ls='-.',color='red',linewidth=1,label='a|0011> + b |0022> + c |0033>')

#ax.plot(U_vec,E_1,ls='-',color='black',linewidth=3,label='|0022>')
#ax.plot(U_vec,E_2,ls='-.',label='|0033>')
#ax.plot(U_vec,E_3,ls='-.',label='|1122>')
#ax.plot(U_vec,E_4,ls=':',label='|1133>')




ax.plot(U_vec,E[:,8], ls='-', color='red', linewidth=lw, label=r'Variational state and effective Hamiltonian')

ax.axis('tight')

ax.set(ylabel='Energy (arb. units)', xlabel='U / t')
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












