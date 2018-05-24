# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 17:07:09 2016

@author: mschueler
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import scipy.sparse as sp

from scipy.optimize import minimize
from scipy.linalg import expm

from plot_scripts import plot_font_size

import pdb

import sys
sys.path.append("../../")
#import pythonModules.ustarCalculation as uStar

import edvaraux.newHelpersMulOrb as hel
import edvaraux.uMatrix as uMat
import edvaraux.newHartreeFock as harFock
import edvaraux.newIO as io

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plot_font_size('poster')

cfg = io.readConfig('parameters.in')
cfg['dateString'] = time.strftime('%a%d%b%Y_%H-%M-%S')
par = hel.makeParamsClass(cfg)

par.beta = 20
par.Nbath = 0 
par.NimpOrbs = 4
par.epsBath = np.array([0.0,0.0,0.0,0.0,0.0])
par.NfStates = 2 * (par.Nbath * par.NimpOrbs) + par.NimpOrbs * 2
nameStrings = hel.makeNameStrings(par) 
print(nameStrings)
ham, blockCar = hel.findAndSetBlockCar(cfg,par,nameStrings)

spStr = ['_up', '_dn']

#uVec = np.arange(0.0,20.0,0.2)
uVec = np.linspace(start=0.0, stop=8.0, num=100)
#uVec = np.array([7.6])
#betaVec = np.array([20.0])

par.beta = 20.0

Energy = np.zeros(uVec.size)
Energy_var = np.zeros(uVec.size)
Energy_gutzwiller = np.zeros(uVec.size)
Energy_baeriswyl = np.zeros(uVec.size)



print('\n{}\n'.format(('-')*70))


v = np.max(ham['diagblk_dim'])
ind = np.argmax(ham['diagblk_dim'])

eig_vecs_save = np.zeros((v, uVec.size))
eig_vals_save = np.zeros((v, uVec.size))



def get_states_Full(oper, ind_0, ind_1):
    oper_diag_N4 = np.diag(oper)[ind_0:ind_1]
    oper_diag_N4_nonzero = np.nonzero(oper_diag_N4)
    return oper_diag_N4_nonzero[0]

def get_states_N4(oper):
    oper_diag_N4 = np.diag(oper)
    oper_diag_N4_nonzero = np.nonzero(oper_diag_N4)
    return oper_diag_N4_nonzero[0]

def sparse2dense_renk(matrix_sparse):
    dummy = matrix_sparse.tocoo()
    matrix_dense = dummy.toarray()
    return matrix_dense


ind0, ind1 = ham['diagblk_ind'][ind], ham['diagblk_ind'][ind+1]

oper_single = np.eye(256)
for i in range(4):
    oper_site = np.zeros((256,256))
    for s in spStr[::-1]:
        oper_str = 'imp' + str(i+1) + s
        oper_N = ham['oper']['+' + oper_str] * ham['oper']['-' + oper_str]
        
        oper_N_dense = sparse2dense_renk(oper_N)
        oper_site += oper_N_dense
    
    oper_single *= oper_site


oper_double = []
for i in range(par.NimpOrbs):
    oper_i_double = np.eye(256)
    for s in spStr:
        oper_str = 'imp' + str(i+1) + s
        oper_i_double *= ham['oper']['+' + oper_str] * ham['oper']['-' + oper_str]
    oper_double.append(oper_i_double[ind0:ind1,ind0:ind1])




oper_2double = np.zeros((36,36))
for i in range(par.NimpOrbs-1):
    for j in range(i+1, par.NimpOrbs):
        oper_2double += oper_double[i] * oper_double[j]



single_states_half = get_states_Full(oper_single, ind0, ind1)
double_x2_states_half = get_states_N4(oper_2double)

#print(single_states_half, type(single_states_half))
#print(double_x2_states_half)

print('\n{}\n'.format(('-')*70))


def get_E(x, Phi_list, H):
    Psi = np.dot(x, Phi_list)
    return np.min(np.dot(Psi, H * Psi))
x_0 = np.ones(2)
cons = {'type': 'eq', 'fun': lambda x: np.dot(x,x) - 1}



#######################################     Gutzwiller      ########################################

def oper_Gutzwiller(g):
    oper_G = np.eye(36)
    for oper in oper_double:
        oper_G *= np.eye(36) - (1 - g) * oper
    return oper_G
        

def get_E_Gutzwiller(x, Psi_0, H):
    oper_G = oper_Gutzwiller(x)
    Psi_G = np.dot(oper_G, Psi_0)
    
    Psi_G /= np.sqrt(np.dot(Psi_G, Psi_G))
    return np.min(np.dot(Psi_G, H * Psi_G))
#cons_G = {'type': 'ineq', 'fun': lambda x: np.dot(x,x) - 1}
    



#######################################         Ht         ########################################

t = 1.0 if par.NimpOrbs > 2.0 else 0.5
Ht = 0.0*copy.copy(ham['fullLocal'])

for ii in range(par.NimpOrbs):
    jj = (ii+1)%par.NimpOrbs
    for iS in range(2):
        Ht += t * (
            ham['oper']['+imp'+str(ii+1)+spStr[iS]] * ham['oper']['-imp'+str(jj+1)+spStr[iS]] + 
            ham['oper']['+imp'+str(jj+1)+spStr[iS]] * ham['oper']['-imp'+str(ii+1)+spStr[iS]] )


#######################################    Hartree-Fock    ########################################

(eigVals_0, eigVecs_0) = hel.diagonalize_blocks(
        Ht, ham['diagblk_qnum'], ham['diagblk_dim'], ham['diagblk_ind'], 'full', cfg['ed'])

#print(eigVals_0[ind])
Ht_N4 = Ht[ind0:ind1, ind0:ind1]
Phi_HF_0 = eigVecs_0[ind][:,0]
Phi_HF_0 /= np.sqrt(np.dot(Phi_HF_0, Phi_HF_0))

Ht_N4_dense = sparse2dense_renk(Ht_N4)


#######################################     Baeriswyl      ########################################

def get_E_Baeriswyl(x, Phi_0, H):
    
    oper_B = expm(x * Ht_N4_dense)    
    Psi_B = np.dot(oper_B, Phi_0)    
    
    Psi_B /= np.sqrt(np.dot(Psi_B, Psi_B))
    return np.min(np.dot(Psi_B, H * Psi_B))



#######################################################################################################
#######################################    Main Calculation    ########################################
#######################################################################################################



for iU in range(uVec.size):
    par.mu = 1/2 * uVec[iU]
    
    HU = 0.0*copy.copy(ham['fullLocal'])
    
    for ii in range(par.NimpOrbs):
        HU += uVec[iU] * (
            ham['oper']['+imp'+str(ii+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up'] * 
            ham['oper']['+imp'+str(ii+1)+'_dn'] * ham['oper']['-imp'+str(ii+1)+'_dn'] )
    
    H = Ht + HU
    
    ################################################################################################
    
    H_N4 = H[ind0:ind1, ind0:ind1]
    
    ################################################################################################
    
    # Our approach
    Phi_0 = np.zeros(256)
    Phi_0[single_states_half+ind0] = 1
    Phi_0 /= np.sqrt(np.dot(Phi_0, Phi_0))
    
    Phi_1 = Ht * Phi_0
    Phi_1 /= np.sqrt(np.dot(Phi_1, Phi_1))
    
    res_0 = minimize(get_E, [1,1], args=([Phi_0, Phi_1], H), method='SLSQP',constraints=cons)
    Energy_var[iU] = res_0.fun
    ################################################################################################
    
    # Gutzwiller
    res_G = minimize(get_E_Gutzwiller, 1, args=(Phi_HF_0, H_N4), method='SLSQP',bounds=((0,1),))
    Energy_gutzwiller[iU] = res_G.fun
    ################################################################################################
    
    # Baeriswyl
    Psi_inf_N4 = np.zeros(36)
    Psi_inf_N4[single_states_half] = 1
    res_B = minimize(get_E_Baeriswyl, 1, args=(Psi_inf_N4, H_N4), method='SLSQP')    
    Energy_baeriswyl[iU] = res_B.fun    
    ################################################################################################
    
    # Malte
    eigVals,eigVecs = hel.diagonalize_blocks(H,ham['diagblk_qnum'],ham['diagblk_dim'],ham['diagblk_ind'],'full',cfg['ed'])
    
    eig_vals_save[:,iU] = np.array(eigVals[ind])
    eig_vecs_save[:,iU] = eigVecs[ind][:,0]
    
    
    ZZ1Til,ZZ1BlockTil,Phi,NG,E0,eigVecs,thermo,minIndex,E0Index = hel.partitionFunction(par,ham,eigVals,eigVecs,cfg)
    
    
#    for jj in range(par.NimpOrbs//2+1):
#        a=hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,
#                                ham['oper']['+imp'+str(1)+'_up'] * ham['oper']['-imp'+str(1)+'_up']*ham['oper']['+imp'+str(jj+1)+'_dn'] * ham['oper']['-imp'+str(jj+1)+'_dn'],cfg,par,thermo)
#        #a=hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+imp'+str(ii+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up'],cfg,par,thermo)
#    
#        a+=hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,
#                                 ham['oper']['+imp'+str(1)+'_up'] * ham['oper']['-imp'+str(1)+'_up']*ham['oper']['+imp'+str(jj+1)+'_up'] * ham['oper']['-imp'+str(jj+1)+'_up'],cfg,par,thermo)
#        double[iB,iU,jj] = a   
    
    Energy[iU] = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,
                                        H,cfg,par,thermo)
    
    



fig, ax = plt.subplots()


ax.plot(uVec, Energy, label='exact')
ax.plot(uVec[1:], Energy_var[1:], label=r'$\alpha \cdot$ single + $\beta \cdot H_t \cdot $ single')
ax.plot(uVec, Energy_gutzwiller,'--', label=r'Gutzwiller')
ax.plot(uVec, Energy_baeriswyl,'--', label=r'Baeriswyl')


ax.set(title='Hubbard 4 site / half filling - variation',
       xlabel=r'$\frac{U}{t}$', ylabel=r'Energy')

ax.legend(loc='best')

plt.show()


#for i in range(v):
#    ax.plot(uVec, eig_vals_save[i,:])
#
#plt.show()
#    
#for x in eig_vecs_save[:,-1]:
#    print(x)


#for x in ham['oper']:
#    print(x)

























