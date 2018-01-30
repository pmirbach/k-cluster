#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:32:14 2018

@author: pmirbach
"""

import numpy as np
from matplotlib import pyplot as plt

import time
#import copy
#import pickle
#import scipy.sparse as sp

import edvaraux.newHelpersMulOrb as hel
##import edvaraux.uMatrix as uMat
##import edvaraux.newHartreeFock as harFock
import edvaraux.newIO as io

cfg = io.readConfig()
cfg['dateString'] = time.strftime('%a%d%b%Y_%H-%M-%S')
par = hel.makeParamsClass(cfg)

par.beta=200
par.Nbath = 0 
par.NimpOrbs = 2
par.epsBath = np.array([0.0,0.0,0.0,0.0])
par.NfStates =2*(par.Nbath*par.NimpOrbs)+par.NimpOrbs*2
nameStrings = hel.makeNameStrings(par) 
print(nameStrings)
ham, blockCar = hel.findAndSetBlockCar(cfg,par,nameStrings)

spStr = ['_up', '_dn']


DeltaU = 0.05
uVec = np.arange(0.0,10,DeltaU)
betaVec = np.array([10.0])


exp_hop = np.zeros((betaVec.size,uVec.size,2*par.NimpOrbs))
single = np.zeros((betaVec.size,uVec.size,par.NimpOrbs,2))
double = np.zeros((betaVec.size,uVec.size,par.NimpOrbs))
Energy = np.zeros((betaVec.size,uVec.size))
Energy_test = np.zeros((betaVec.size,uVec.size))


t = 1.0 if par.NimpOrbs > 2 else 0.5
    


for iU in range(uVec.size):
    par.mu = uVec[iU]*0.5
    for iB in range(betaVec.size):
        H = 0.0*copy.copy(ham['fullLocal'])
        par.beta = betaVec[iB]
        
        for ii in range(par.NimpOrbs):
            H  = H + uVec[iU]*(
                ham['oper']['+imp'+str(ii+1)+'_up']*
                ham['oper']['-imp'+str(ii+1)+'_up']*
                ham['oper']['+imp'+str(ii+1)+'_dn']*
                ham['oper']['-imp'+str(ii+1)+'_dn']
                )
        for ii in range(par.NimpOrbs):
            jj = (ii+1)%par.NimpOrbs
            for iS in range(2):
                H = H + t*(
                    ham['oper']['+imp'+str(ii+1)+spStr[iS]] * ham['oper']['-imp'+str(jj+1)+spStr[iS]]
                    +
                    ham['oper']['+imp'+str(jj+1)+spStr[iS]] * ham['oper']['-imp'+str(ii+1)+spStr[iS]]
                    )

        eigVals,eigVecs = hel.diagonalize_blocks(H,ham['diagblk_qnum'],ham['diagblk_dim'],ham['diagblk_ind'],'full',cfg['ed'])
        ZZ1Til,ZZ1BlockTil,Phi,NG,E0,eigVecs,thermo,minIndex,E0Index = hel.partitionFunction(par,ham,eigVals,eigVecs,cfg)
        
        

        for jj in range(par.NimpOrbs):
            a = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,
                                      ham['oper']['+imp'+str(jj+1)+'_up'] * ham['oper']['-imp'+str(jj+1)+'_up'] * 
                                      ham['oper']['+imp'+str(jj+1)+'_dn'] * ham['oper']['-imp'+str(jj+1)+'_dn'],
                                      cfg,par,thermo)
            double[iB,iU,jj] = a
            
            for iS in range(2):
                b = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,
                                          ham['oper']['+imp'+str(jj+1)+spStr[iS]] * 
                                          ham['oper']['-imp'+str(jj+1)+spStr[iS]],
                                          cfg,par,thermo)
                single[iB,iU,jj,iS] = b

            ii = (jj + 1) % par.NimpOrbs
            c1 = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,
                                       ham['oper']['+imp'+str(ii+1)+'_up'] * ham['oper']['-imp'+str(jj+1)+'_up'],
                                       cfg,par,thermo)
            c2 = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,
                                       ham['oper']['+imp'+str(jj+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up'],
                                       cfg,par,thermo)
            exp_hop[iB,iU,ii] = c1
            exp_hop[iB,iU,ii+par.NimpOrbs] = c2
            
        
        Energy[iB,iU]=hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,
                                            H,cfg,par,thermo)
        Energy_test[iB,iU] = 2 * t * np.sum(exp_hop[iB,iU,:]) + uVec[iU] * np.sum(double[iB,iU,:])
        
        
        Energy_cheat = Energy * 2

    np.save('doubleOcc',double)
	

double = double[:32,:,:]
single = single[:32,:,:,:]
betaVec = betaVec[:32]

single = np.round(single, decimals=3)



#plt.plot(uVec,exp_hop[0,:,0,0])
#plt.show()

#==============================================================================
# plt.figure(1)
# for ii in range(par.NimpOrbs):
#     nr_subplot = int(str(par.NimpOrbs) + "1" + str(ii+1))
#     plt.subplot(nr_subplot)
#     plt.plot(uVec,single[0,:,ii,0])
#     plt.plot(uVec,single[0,:,ii,1],'--')
# 
# plt.figure(2)
# for ii in range(par.NimpOrbs):
#     nr_subplot = int(str(par.NimpOrbs) + "1" + str(ii+1))
#     plt.subplot(nr_subplot)
#     plt.plot(uVec,double[0,:,ii])
#==============================================================================

#==============================================================================
# f, axarr = plt.subplots(2*par.NimpOrbs)
# for ii in range(2*par.NimpOrbs):
#     for iS in range(2):
#         axarr[ii].plot(uVec,exp_hop[0,:,ii])
#==============================================================================



#plt.plot(uVec,Energy[0,:])
#plt.plot(uVec,Energy_test[0,:],'--')
#plt.plot(uVec,Energy_cheat[0,:],'--')

#plt.show()




#==============================================================================
# f = open("Hub4_ed.pkl","rb")
# (uVec_ed, Energy_ed) = pickle.load(f)
# f.close()
# 
# plt.plot(uVec_ed,Energy_ed)
# plt.plot(uVec,Energy_cheat[0,:],'--')
# 
# plt.show()
#==============================================================================






#==============================================================================
#==============================================================================
# # fh = open("Hub2_SC.pkl","bw")
# # pickle_object = (uVec, Energy_cheat[0,:])
# # pickle.dump(pickle_object,fh)
# # fh.close()
#==============================================================================
#==============================================================================

#==============================================================================
#==============================================================================
# # fh = open("Hub4_ed.pkl","bw")
# # pickle_object = (uVec, Energy[0,:])
# # pickle.dump(pickle_object,fh)
# # fh.close()
#==============================================================================
#==============================================================================