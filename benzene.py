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

import sys
sys.path.append("../../")
#import pythonModules.ustarCalculation as uStar

import edvaraux.newHelpersMulOrb as hel
import edvaraux.uMatrix as uMat
import edvaraux.newHartreeFock as harFock
import edvaraux.newIO as io

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
uVec = np.linspace(start=0.0, stop=8.0, num=50)
#uVec = np.array([2.0])
#betaVec = np.array([20.0])

par.beta = 20.0

Energy = np.zeros(uVec.size)


print()
print('-------------------------')
print()


v = np.max(ham['diagblk_dim'])
ind = np.argmax(ham['diagblk_dim'])

eig_vecs_save = np.zeros((v, v, uVec.size))



#exit()


t = 1.0 if par.NimpOrbs > 2.0 else 0.5


for iU in range(uVec.size):
    par.mu = 1/2 * uVec[iU]
    
    H = 0.0*copy.copy(ham['fullLocal'])
    
    for ii in range(par.NimpOrbs):
        H  = H + uVec[iU] * (
            ham['oper']['+imp'+str(ii+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up'] * 
            ham['oper']['+imp'+str(ii+1)+'_dn'] * ham['oper']['-imp'+str(ii+1)+'_dn'] )

    for ii in range(par.NimpOrbs):
        jj = (ii+1)%par.NimpOrbs
        for iS in range(2):
            H = H + t * (
                ham['oper']['+imp'+str(ii+1)+spStr[iS]] * ham['oper']['-imp'+str(jj+1)+spStr[iS]] + 
                ham['oper']['+imp'+str(jj+1)+spStr[iS]] * ham['oper']['-imp'+str(ii+1)+spStr[iS]] )
    
        
    eigVals,eigVecs = hel.diagonalize_blocks(H,ham['diagblk_qnum'],ham['diagblk_dim'],ham['diagblk_ind'],'full',cfg['ed'])
    eig_vecs_save[:,:,iU] = np.array(eigVecs[ind])
    
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

#x = ham['oper']['+imp1_up']
#
#print(type(x))
#x2 = x.todense()

#eig_test = eig_vecs_save[:,0,0]
#print(eig_test, np.dot(eig_test, eig_test))
#
#eig_test = eig_vecs_save[0,:,0]
#print(eig_test, np.dot(eig_test, eig_test))


print(np.transpose(eig_vecs_save[0,:,-1]))
exit()

eig_plot = np.transpose(eig_vecs_save[0,:,-1])

fig, ax = plt.subplots()

ax.plot(eig_plot)
#for i in range(v):
#    y = np.transpose((eig_plot[i,:]))
#    ax.plot(uVec, np.sort(y),'--')


#plt.plot(uVec,Energy)
plt.show()
    







