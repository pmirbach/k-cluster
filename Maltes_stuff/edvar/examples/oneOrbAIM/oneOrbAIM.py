
import sys
import os

# add the current directory to python path, so we can load the parametersFile
sys.path.insert(0,os.getcwd())

import matplotlib.pylab as plt
import numpy as np
import copy
import time
import scipy.sparse as sp


import edvaraux.newHelpersMulOrb as hel
import edvaraux.uMatrix as uMat
import edvaraux.newHartreeFock as harFock
import edvaraux.newIO as io


cfg = io.readConfig()
cfg['dateString'] = time.strftime('%a%d%b%Y_%H-%M-%S')


par = hel.makeParamsClass(cfg)

par.Nbath = cfg['aim']['Nbath']
par.vBath = np.zeros(shape=(par.NimpOrbs,par.Nbath))
par.epsBath = np.zeros(shape=(par.NimpOrbs,par.Nbath))
for iO in range(par.NimpOrbs):
    par.vBath[iO,:] = cfg['aim']['vBath'+str(iO+1)]
    par.epsBath[iO,:] = cfg['aim']['epsBath'+str(iO+1)]
    
par.NfStates =2*(par.Nbath*par.NimpOrbs)+par.NimpOrbs*2

par.uMatrix = uMat.uMatrixWrapper(par,cfg)

nameStrings = hel.makeNameStrings(par)

ham, blockCar = hel.findAndSetBlockCar(cfg,par,nameStrings)
start = time.time()
print(cfg['ed']['method'])
print('setting H... ',end='',flush=True)
Htwo = hel.set_ham_pure_twoPart(par,ham)
H = hel.set_ham_pure(par,ham)
print('took '+ str(time.time() - start)+ ' sec',flush=True)
start = time.time()
print('solving H... ',end='',flush=True)

eigVals,eigVecs = hel.diagonalize_blocks(H,ham['diagblk_qnum'],ham['diagblk_dim'],ham['diagblk_ind'],'sparse',cfg['ed'])
print('took '+ str(time.time() - start)+ ' sec',flush=True)

start = time.time()
print('calculating expectation values... ',end='',flush=True)

ZZ1Til,ZZ1BlockTil,Phi,NG,E0,eigVecs,thermo,minIndex,E0Index = hel.partitionFunction(par,ham,eigVals,eigVecs,cfg)

# calculation of <i|c^+c|j>
     
den1 = np.zeros(shape=(2,par.NimpOrbs,par.Nbath))
for iO in range(par.NimpOrbs):
    for iB in range(par.Nbath):
        # up
        ham['bathUp'][iO][iB][0:1,0:1]
        den1[0,iO,iB] = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['bathUp'][iO][iB],cfg,par,thermo)
        den1[1,iO,iB] = den1[0,iO,iB]

        
# calculation of <i|c^+d|j>
cd = np.zeros(shape=(4,par.NimpOrbs,par.Nbath)) 

for iO in range(par.NimpOrbs):
    for iB in range(par.Nbath):
        # UpUp
        cd[0,iO,iB] = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['hybUp'][iO][iB],cfg,par,thermo)
        # UpDn
        cd[1,iO,iB] = 0.0
        # DnUp
        cd[2,iO,iB] = 0.0
        # DnDn
        cd[3,iO,iB] = cd[0,iO,iB]

# calculation of <i|d^+d|j> 
denImp = np.zeros(shape=(2,par.NimpOrbs))
for ii in range(par.NimpOrbs):
    # up
    denImp[0,ii] = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+imp'+str(ii+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up'],cfg,par,thermo)
    denImp[1,ii] = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+imp'+str(ii+1)+'_dn'] * ham['oper']['-imp'+str(ii+1)+'_dn'],cfg,par,thermo)

double =(hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+imp'+str(1)+'_up'] * ham['oper']['-imp'+str(1)+'_up']*ham['oper']['+imp'+str(1)+'_dn'] * ham['oper']['-imp'+str(1)+'_dn'],cfg,par,thermo))
energy = hel.calcExpecValue(eigVecs,ham['diagblk_ind'],ZZ1BlockTil,ZZ1Til,H,par)
energyTwo = hel.calcExpecValue(eigVecs,ham['diagblk_ind'],ZZ1BlockTil,ZZ1Til,Htwo,par)
print('energyOne',energy-energyTwo)
print('energyTwo',energyTwo)
print ('energy',energy)

print('took '+ str(time.time() - start)+ ' sec',flush=True)


DenMat = np.zeros((par.NimpOrbs+par.Nbath,par.NimpOrbs+par.Nbath))
for ii in range(par.NimpOrbs+par.Nbath):
    for jj in range(par.NimpOrbs+par.Nbath):
        DenMat[ii,jj] = hel.calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+'+str(ii)+'_up'] * ham['oper']['-'+str(jj)+'_up'],cfg,par,thermo)


print('nAt',denImp[:,0])
print('nBath',den1[0,:,:])
print('cd',cd[0,:,:])
print('Phi',Phi)
print('<nn>',double)

# possibilty to save. therfor write anything you want into the dictionary
#out = dict()
#io.saveResultNew(cfg,out)



   
