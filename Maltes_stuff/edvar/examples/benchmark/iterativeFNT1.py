#! /usr/bin/env python

import sys
import os

# add the current directory to python path, so we can load the parametersFile
sys.path.insert(0,os.getcwd())

import numpy as np
import copy
import time
import scipy.sparse as sp

import matplotlib.pyplot as plt


import edvaraux.newHelpersMulOrb as hel
import edvaraux.newIO as io
import edvaraux.uMatrix as uMat
import edvaraux.newHartreeFock as harFock
import pickle

cfg = io.readConfig()
cfg['dateString'] = time.strftime('%a%d%b%Y_%H-%M-%S')

par = hel.makeParamsClass(cfg)
parEff = hel.makeParamsClass(cfg)

parEff.Nbath = 1
parEff.NfStates =2*(parEff.Nbath*parEff.NimpOrbs)+parEff.NimpOrbs*2
parEff.vBath = np.zeros(shape=(parEff.NimpOrbs,parEff.Nbath))
parEff.epsBath = np.zeros(shape=(parEff.NimpOrbs,parEff.Nbath))


par.Nbath = cfg['aim']['Nbath']
par.vBath = np.zeros(shape=(par.NimpOrbs,par.Nbath))
par.epsBath = np.zeros(shape=(par.NimpOrbs,par.Nbath))
for iO in range(par.NimpOrbs):
    par.vBath[iO,:] = cfg['aim']['vBath'+str(iO+1)]
    par.epsBath[iO,:] = cfg['aim']['epsBath'+str(iO+1)]

start = time.time()

if cfg['aim']['readRealMat'] or cfg['aim']['epsBathRange']:
    cfg,par,parEff = hel.setRealPar(cfg,par,parEff)
else:
    par.Nbath = cfg['aim']['Nbath']
    par.vBath = np.zeros(shape=(par.NimpOrbs,par.Nbath))
    par.epsBath = np.zeros(shape=(par.NimpOrbs,par.Nbath))
    for iO in range(par.NimpOrbs):
        par.vBath[iO,:] = cfg['aim']['vBath'+str(iO+1)]
        par.epsBath[iO,:] = cfg['aim']['epsBath'+str(iO+1)]
#exit()
par.Bz = cfg['aim']['Bz']
# calculate uMatrix
parEff.uMatrix = uMat.uMatrixWrapper(parEff,cfg)
par.uMatrix = parEff.uMatrix

print(par.epsImp)


par.mu = 0.0
parEff.mu = 0.0

par = io.checkInput(par,parEff,cfg)
nameStrings = hel.makeNameStrings(parEff)        

hFSol = harFock.solveHartreeFock(par,cfg)


#np.savetxt('hamiltonFromHartreeFock.dat',hFSol['hamilLoc'])
#print(hFSol['denMat'][:5,:5])
#print(hFSol['hamilLoc'][:5,:5])
#exit()
#harFock.solveHartreeFockUnres(par,cfg)
#exit()

# get first guess by cafarell krauth
print(parEff.numDeg)
print(parEff.degeneracy)
point = cfg['algo']['firstGuess']
if cfg['algo']['firstGuessCafKrauth']:
    # overwrite first guess by caf. krauth
    point = hel.firstGuessCafKrauth(par,cfg)

ham, blockCar = hel.findAndSetBlockCar(cfg,parEff,nameStrings)



startCoeffsD, startCoeffsC1 = harFock.firstGuessHFSubSpace(par)

# projection from HF-Hamiltonian defines first Guess for parameters of
# correlated subspace:
onePartHamilCorrSub,_,twoPartHamilCorrSub, twoPartHamilCorrSubAll = harFock.corrSubHamiltonian(cfg,par,hFSol,startCoeffsD,startCoeffsC1)

   



hf = hFSol
    
if (cfg['algo']['optOnlyBath'] == False) and (cfg['algo']['firstGuessCafKrauth']):
    pointCopy = copy.copy(point)
    #if par.NimpOrbs>1:
    #    par.maxIterOuter = 0
    point = np.zeros(shape=(3*par.numDeg))
    for iD in range(par.numDeg):
        point[iD] = par.epsImp[par.degeneracy[iD][0]]
        point[iD+par.numDeg] = pointCopy[iD]
        point[iD+2*par.numDeg] =  pointCopy[iD+par.numDeg]
print('\nfirst point print')
print(point)
out = hel.opt(cfg,point,ham,par,parEff,blockCar,hf)

cfg['ed']['excited'] = 0
w2T,dos2T = hel.calcEffectiveSpectraBeta(par,ham,cfg,hf)

print('time for minim',  time.time()- start)
optPoint = out['xopt'].copy()
print('opt point', optPoint)
_, coeffD, coeffC1 = io.getRecentOnePartBasis(cfg)
print('eff imp: impMix:',coeffD[0,0])


# as it is implemented now: make sure optPoint is the same point as the one hel.calcPhiTildeED did the last calculation
completeDenMatLocBasis, energy, twoPartLocUpUp, twoPartLocUpDn, coeffs, sSquare, sSquareB = hel.calcUpdatedObservables(optPoint,ham,cfg,par,parEff,blockCar,hf)
np.savetxt('completeTrafoMat.txt',coeffs)
if False:
    print('electrons in the system',np.trace(completeDenMatLocBasis))
    nGS = np.round(np.trace(completeDenMatLocBasis))
    nGS = 2
    
    cfg['ed']['excited'] = +1
    cfg['ed']['Nmin'] = nGS+cfg['ed']['excited']
    cfg['ed']['Nmax'] = nGS+cfg['ed']['excited']
    
    hamP1, blockCarP1 = hel.findAndSetBlockCar(cfg,parEff,nameStrings)
    
    outP1 = hel.opt(cfg,point,hamP1,par,parEff,blockCarP1,hf)
    
    cfg['ed']['excited'] = -1
    cfg['ed']['Nmin'] = nGS+cfg['ed']['excited']
    cfg['ed']['Nmax'] = nGS+cfg['ed']['excited']
    
    
    hamM1, blockCarM1 = hel.findAndSetBlockCar(cfg,parEff,nameStrings)
    outM1 = hel.opt(cfg,point,hamM1,par,parEff,blockCarM1,hf)
#w,dos = hel.calcOriginalSpectra(par,ham,cfg,edSol['eigVals'],edSol['eigVecs'])
#w2T,dos2T = hel.calcEffectiveSpectraBeta(par,ham,cfg,hf)
#w2TM,dos2TM = hel.calcEffectiveSpectraBetaMatrix(par,ham,cfg,hf)
#w2TM,dos2TM_minus = hel.calcEffectiveSpectraBetaMatrixMinus(par,ham,cfg,hf)

numOPS = par.NimpOrbs + par.NimpOrbs*par.Nbath
epsdMatrix=  np.diag(par.epsImp)
hamNonInt=harFock.onePartMatrix(par.epsBath,par.vBath,epsdMatrix,par.Nbath)

if False:
    energyGM = np.sum(
 (w2TM[:,np.newaxis,np.newaxis] * np.eye(numOPS)[np.newaxis,:,:]
     + hamNonInt[np.newaxis,:,:] ) * dos2TM_minus 
                 )*(w2TM[1]-w2TM[0])

    print('energy Gal. Migdal',energyGM)

#w2TCl,dos2TCl = hel.calcEffectiveSpectraBetaClever(par,ham,hamP1,hamM1,cfg,hf,nGS)

#print(outM1['xopt'])
#print(outP1['xopt'])
#print(out['xopt'])
if False:
    plt.figure(1)
    plt.clf()
    #plt.plot(w2T,dos2T,'k')
    plt.plot(w2TM,dos2TM[:,0,0],'--c')
    plt.plot(w2TM,dos2TM[:,4,4],'--r')
    plt.plot(w2TM,dos2TM[:,0,1],'--k')
    #plt.plot(w2TCl,dos2TCl,'r')
    dataOrig = np.genfromtxt('../oneOrbAIM/dosT09Beta3200.dat')
    dataOrig44 = np.genfromtxt('../oneOrbAIM/dosT0.90Beta3200_4_4.dat')
    dataOrig01 = np.genfromtxt('../oneOrbAIM/dosT0.90Beta3200_0_1.dat')
    plt.plot(dataOrig[:,0],dataOrig[:,1],'c')
    plt.plot(dataOrig44[:,0],dataOrig44[:,1],'r')
    plt.plot(dataOrig01[:,0],dataOrig01[:,1],'k')
    plt.show()

#w2T,dos2T = hel.calcEffectiveSpectraBeta(par,ham,cfg,hf)
#np.savetxt('dos.dat',np.vstack((w,dos)).transpose())
#np.savetxt('dos2.dat',np.vstack((w2,dos2)).transpose())
if cfg['ed']['calcSpectra']:
    saveName = 'dosT09'
    saveName += 'Weight' + str(cfg['algo']['weightFunc'])
    if cfg['algo']['maxIterOuter'] != 0:
        saveName += 'Full'
    if cfg['algo']['optOnlyBath']:
        saveName += 'OnlyBath'
    saveName += 'Beta'+str(cfg['aim']['beta'])
    print(saveName)
    #np.savetxt(saveName+'.dat',np.vstack((w2T,dos2T)).transpose())

for iO in range(par.NimpOrbs):
    coeffC1[iO,:] /= np.sqrt(np.sum(coeffC1[iO,:]**2))
    coeffD[iO,:] /= np.sqrt(np.sum(coeffD[iO,:]**2))

print('d coeffs')
print( coeffD[0,:])
print('c coeffs')
print(coeffC1[0,:])

f = open(cfg['tempFileConstrain'],'rb')
pickDic = pickle.load(f)
f.close()
print('leg coeffs')
print(pickDic['point'])


#plt.figure(1)
#plt.clf()
#np.savetxt('dummy.dat',coeffC1[0,1:])
#ori = np.genfromtxt('dummy.dat')
#plt.plot(par.epsBath[0,:],ori,'rx')
#plt.plot(par.epsBath[0,:],coeffC1[0,1:],'r')
#plt.plot(par.epsBath[0,:],coeffD[0,1:],'b')
#plt.plot(0,coeffD[0,0],'rx')
#plt.plot(0,coeffD[0,0],'bx')
#plt.show()
twoPart = dict()
twoPart['upup'] = twoPartLocUpUp
twoPart['updn'] = twoPartLocUpDn
out['coeffs'] = coeffs
out['coeffsD'] = coeffD
out['coeffsC1'] = coeffC1
if cfg['algo']['noHF'] == False:
    out['HFDenImp'] = hFSol['denMat']
out['densMat'] = completeDenMatLocBasis
out['twoPart'] = twoPart
out['S^2'] = sSquare


if cfg['algo']['noHF'] == False:
    print('<nIMP HF>', np.diag(hFSol['denMat']),np.sum(np.diag(hFSol['denMat'])))

print( '<nImp>', np.diag(completeDenMatLocBasis)[:par.NimpOrbs],2*np.sum(np.diag(completeDenMatLocBasis)[:par.NimpOrbs]))
print( '<S^2>', sSquare, sSquareB)
print('S',-0.5+np.sqrt(0.25+sSquare))
print( '<nn>', twoPart['updn'][0,0,0,0])

io.saveResultNew(cfg,out)

