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

cfg = io.readConfig()
cfg['dateString'] = time.strftime('%a%d%b%Y_%H-%M-%S')
par = hel.makeParamsClass(cfg)
print(par.NimpOrbs)
par.beta = 200
par.Nbath = 0 
par.NimpOrbs = 4
par.epsBath = np.array([0.0,0.0,0.0,0.0,0.0])
par.NfStates = 2*(par.Nbath*par.NimpOrbs)+par.NimpOrbs*2
nameStrings = hel.makeNameStrings(par) 
print(nameStrings)
ham, blockCar = hel.findAndSetBlockCar(cfg,par,nameStrings)

spStr = ['_up', '_dn']

#uVec = np.arange(0.0,5.0,0.1)
uVec = np.array([2.0])
betaVec = np.array([20.0])

par.beta = 20.0

Energy = np.zeros(uVec.size)



for iU in range(uVec.size):
    par.mu = uVec[iU]*0.0
    
    H = 0.0*copy.copy(ham['fullLocal'])
    
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
            H = H + 1.0*(
                ham['oper']['+imp'+str(ii+1)+spStr[iS]]*
                ham['oper']['-imp'+str(jj+1)+spStr[iS]]
                +
                ham['oper']['+imp'+str(jj+1)+spStr[iS]]*
                ham['oper']['-imp'+str(ii+1)+spStr[iS]]
            )
    
#    H2 = H.todense()
        
    eigVals,eigVecs = hel.diagonalize_blocks(H,ham['diagblk_qnum'],ham['diagblk_dim'],ham['diagblk_ind'],'full',cfg['ed'])
    ZZ1Til,ZZ1BlockTil,Phi,NG,E0,eigVecs,thermo,minIndex,E0Index = hel.partitionFunction(par,ham,eigVals,eigVecs,cfg)
#    for jj in range(par.NimpOrbs//2+1):
#        a=hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+imp'+str(1)+'_up'] * ham['oper']['-imp'+str(1)+'_up']*ham['oper']['+imp'+str(jj+1)+'_dn'] * ham['oper']['-imp'+str(jj+1)+'_dn'],cfg,par,thermo)
#        #a=hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+imp'+str(ii+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up'],cfg,par,thermo)
#    
#        a+=hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+imp'+str(1)+'_up'] * ham['oper']['-imp'+str(1)+'_up']*ham['oper']['+imp'+str(jj+1)+'_up'] * ham['oper']['-imp'+str(jj+1)+'_up'],cfg,par,thermo)
#        double[iB,iU,jj] = a
            
    Energy[iU]=hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,
                                        H,cfg,par,thermo)

x = ham['oper']['+imp1_up']

print(type(x))
x2 = x.todense()

#plt.plot(uVec,Energy)
#plt.show()
    
    
    

#fig, ax = plt.subplots()
#cax = plt.imshow(H2, interpolation='nearest')
#cbar = fig.colorbar(cax)
#plt.show()
            
            
#    np.save('doubleOcc',double)
#
#double  = double[:32,:,:]
#betaVec = betaVec[:32]
#doubleDer = np.zeros(double.shape)
#print(double.shape)
#print(uVec.shape)
#for iR in range(double.shape[2]):
#    for iB in range(double.shape[0]):
#        doubleDer[iB,:,iR] = np.gradient(double[iB,:,iR])/np.gradient(uVec)
#  
#plt.plot(betaVec,double[:,5,0],'-r')  
#plt.plot(betaVec,double[:,5,1],'-b')
##plt.plot(uVec,doubleDer[5,:,2])
##plt.plot(uVec,doubleDer[5,:,3])
#plt.show()
#
#
#spexStr = '/home/mschueler/work/QuestCalc/UVData/BenzenePPP.dat';
#
#dataS = np.genfromtxt(spexStr,skip_header=7)
#
#t = dataS[1]
#U0 = dataS[2]/t
#V1 = 2.0*dataS[3]/t
#V2 = 2.0*dataS[4]/t
#V3 = dataS[5]/t
#V = np.array([V1,V2,V3])
#
#solveThis = U0 +np.sum(V[np.newaxis,np.newaxis,:]*doubleDer[:,:,1:],axis=2)/doubleDer[:,:,0] - uVec
#uStarM = np.zeros(betaVec.size)
#
#for iB in range(double.shape[0]):
#    #plt.title(str(betaVec[iB]))
#    #plt.plot(uVec,solveThis[iB,:])
#    #plt.axis((0.0,5.0,0.0,4.0))
#    #plt.show()
#    uStarM[iB] = uStar.linInt(uVec,solveThis[iB,:])
#
#    #
##print(betaVec.shape)
##print(uStarM.shape)
#a = np.vstack((betaVec,uStarM)).transpose()
#print(a.shape)
#np.savetxt('betaBenzeneDataCanonical.dat',a)
##plt.plot(betaVec,uStarM)
##plt.show()