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
#uVec = np.array([7.6])
#betaVec = np.array([20.0])

par.beta = 20.0

Energy = np.zeros(uVec.size)


print('\n{}\n'.format(('-')*70))


v = np.max(ham['diagblk_dim'])
ind = np.argmax(ham['diagblk_dim'])

eig_vecs_save = np.zeros((v, uVec.size))
eig_vals_save = np.zeros((v, uVec.size))


#for x,y in ham['oper'].items():
#    print(x, type(y))
#    
#    print(y.nnz)
#    
##    rows, cols = y.nonzero()
##    print(rows, cols)
##    print(y.shape)
#    print(dir(y))
#    print(y.format)
#    print(y.data)
#    
#    z = y.tocoo()
#    z2 = z.toarray()
#    
#    print(z2)
#    
#    plt.imshow(z2)
#    plt.show()
#    
#    
#    break

#a = np.arange(256)
#print(a[110:146])




ind0, ind1 = ham['diagblk_ind'][ind], ham['diagblk_ind'][ind+1]


asdf = ham['oper']['+imp1_up'] * ham['oper']['-imp1_up']
asdf2 = asdf.tocoo()
asdf3 = asdf2.toarray()

#ham['oper']['+imp'+str(ii+1)+'_up']

plt.imshow(asdf3[ind0:ind1,ind0:ind1])
plt.show()

#fig, axes = plt.subplots(nrows=2, ncols=2)
#
#fig_manager = plt.get_current_fig_manager()
#if hasattr(fig_manager, 'window'):
#    fig_manager.window.showMaximized()
#
##print(axes)
#for i, ax in enumerate(fig.axes):
#    ax.set(title=i)
##    axes[i].imshow()
#
#plt.show()
#
#print('\n{}\n'.format(('-')*70))
    
exit()




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

#ax.plot(eig_plot)
#for i in range(v):
#    y = np.transpose((eig_plot[i,:]))
#    ax.plot(uVec, np.sort(y),'--')


#plt.plot(uVec,Energy)


#for i in range(v):
#    ax.plot(uVec, eig_vals_save[i,:])
#
#plt.show()
#    
for x in eig_vecs_save[:,-1]:
    print(x)


#for x in ham['oper']:
#    print(x)

























