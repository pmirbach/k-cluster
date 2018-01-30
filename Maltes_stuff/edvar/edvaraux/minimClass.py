# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:18:08 2016

@author: mschueler
"""

import numpy as np
import matplotlib.pyplot as plt
import edvaraux.newIO as io
import edvaraux.uMatrix as uMat
import edvaraux.newHartreeFock as harFock
import edvaraux.newHelpersMulOrb as hel
from scipy import optimize
import time
import copy
class minimizer:
    
    def __init__(self,cfg):
        parameters = hel.makeParamsClass(cfg)     
        self.globalMinim = 1e100
        self.globalMinimPoint = None
        self.point = None
        self.par = copy.copy(parameters)
        self.parEff = copy.copy(parameters)
        self.localHamiltonian = None
        self.cfg = cfg
        self.ham = None
        self.blockCar = None
        self.cD = None
        self.cC = None
        self.pointBasis = None
        self.numParams = 0
        self.par.uMatrix = uMat.uMatrixWrapper(self.parEff,cfg)
        self.parEff.uMatrix = copy.copy(self.par.uMatrix)
        self.parEff.epsImp = copy.copy(self.par.epsImp)

        # initialize effective bath and uMatrix with nan
        self.parEff.uMatrix *= np.nan
        self.parEff.epsImp *= np.nan        

        # set number of free parameters of the effective Hamiltonian
        if cfg['algo']['optU']:
            if self.par.NimpOrbs == 1:
                # only U is varied
                self.numParams += 1
            else:
                # U and J are varied
                self.numParams += 2
        if cfg['algo']['optEpsD']:
            self.numParams += self.par.numDeg
        if cfg['algo']['optV']:
            self.numParams += self.par.numDeg
        if cfg['algo']['optEpsBath']:
            self.numParams += self.par.numDeg
        if cfg['algo']['optOnlyBath']:
            if cfg['algo']['fitLegendre']:
                self.pointToVec = hel.pointToVecOnlyBathLegendre
            else:
                self.pointToVec = hel.pointToVecOnlyBathSimple
        else:
            if cfg['algo']['fitLegendre']:
                self.pointToVec = hel.pointToVecOptAllLegendre
            else:
                self.pointToVec = hel.pointToVecOptAllSimple        
        return
        
        
    def setEffectiveBath(self,numStates):
        self.parEff.Nbath = numStates
        self.parEff.NfStates =2*(self.parEff.Nbath*self.parEff.NimpOrbs)+self.parEff.NimpOrbs*2
        self.parEff.vBath = np.zeros(shape=(self.parEff.NimpOrbs,self.parEff.Nbath))*np.nan
        self.parEff.epsBath = np.zeros(shape=(self.parEff.NimpOrbs,self.parEff.Nbath))*np.nan

    def setOriginalBath(self):
        if self.cfg['aim']['readRealMat'] or self.cfg['aim']['epsBathRange']:
            self.cfg,self.par,self.parEff = hel.setRealPar(self.cfg,self.par,self.parEff)
        else:
            self.par.Nbath = self.cfg['aim']['Nbath']
            self.par.vBath = np.zeros(shape=(self.par.NimpOrbs,self.par.Nbath))
            self.par.epsBath = np.zeros(shape=(self.par.NimpOrbs,self.par.Nbath))
            for iO in range(self.par.NimpOrbs):
                self.par.vBath[iO,:] = self.cfg['aim']['vBath'+str(iO+1)]
                self.par.epsBath[iO,:] = self.cfg['aim']['epsBath'+str(iO+1)]        
    
    def manyBodyBasis(self):
        nameStrings = hel.makeNameStrings(self.parEff) 
        self.ham, self.blockCar = hel.findAndSetBlockCar(self.cfg,self.parEff,nameStrings)
    
    def firstGuess(self):
        self.point = self.cfg['algo']['firstGuess']
            
    
    def checkFirstGuess(self):
        if self.numParams != len(self.point):
            raise Exception('vector of parameters does not have the size of number of parameters!')
    
    def checkInput(self):
        
        
        #if (self.cfg['algo']['optV'] == False) and (self.cfg['algo']['optEpsBath'] == False) and (self.cfg['algo']['firstGuessCafKrauth']):
        #    raise Exception('not optimizing V or epsBath, so no need for caf-krauth first guess!')
        
        if self.par.numDeg == 0 or self.parEff.numDeg == 0:    
            raise Exception('parameter numDeg = 0 depreciated')
        
        print('checking input',self.cfg['aim']['overRideV_Deg_Check'])
        if (self.par.vBath.shape[1] != self.par.Nbath) or self.par.vBath.shape[0] != self.par.NimpOrbs:
            raise Exception('vbath dimension does not fit to number of baths or orbitals')
        if (self.par.epsBath.shape[1] != self.par.Nbath) or self.par.epsBath.shape[0] != self.par.NimpOrbs:
            raise Exception('epsbath dimension does not fit to number of baths or orbitals')
        if (self.par.epsImp.size != self.par.NimpOrbs) or self.par.epsBath.shape[0] != self.par.NimpOrbs:
            raise Exception('epsImp dimension does not fit to number of baths or orbitals')
        
        if self.cfg['algo']['fitLegendre']:
            if self.cfg['algo']['legendreOrder'] < 1:
                raise Exception('legendreOrder is 0, but has to be >0')
        try:
            toleranceVCheck=self.cfg['aim']['degeneracyTol']
        except:
            toleranceVCheck = 1e-15
    
    
        if (self.par.numDeg > 0):
            if (self.par.numDeg != len(self.par.degeneracy)):
                raise Exception('number of degeneracies given in list par.degeneracy does not fit not number in par.numDeg')
            # check if mean over orbital energy and hyb is equal to all energies and hybs
            for iD in range(self.par.numDeg):
                meanVals = np.mean(self.par.epsBath[self.par.degeneracy[iD],:],axis=0)
                for iOD in range(len(self.par.degeneracy[iD])):
                    if np.any(np.abs(meanVals - self.par.epsBath[self.par.degeneracy[iD][iOD],:])>1e-15):
                        raise Exception('degenercy of energies not recognized, this check should NOT be overriden')
                meanVals = np.mean(self.par.vBath[self.par.degeneracy[iD],:],axis=0)
                if self.cfg['aim']['overRideV_Deg_Check'] == False:
                    for iOD in range(len(self.par.degeneracy[iD])):    
                        if np.any(np.abs(meanVals - self.par.vBath[self.par.degeneracy[iD][iOD],:])>toleranceVCheck):
                            maxDiscrep = np.max(np.abs(meanVals - self.par.vBath[self.par.degeneracy[iD][iOD],:]))
                            print('maximal discrepeancy from mean value:',maxDiscrep)
                            raise Exception('degenercy of hyb not recognized, maybe override this check..')
                    # to really have degenerate bands, we now use the mean values as input:
                    self.par.vBath[self.par.degeneracy[iD][:],:] = meanVals
                    meanVals = np.mean(self.par.epsImp[self.par.degeneracy[iD]])
                    self.par.epsImp[self.par.degeneracy[iD][:]] = meanVals
                    self.par.epsImp[self.parEff.degeneracy[iD][:]] = meanVals
                else:
                    print('Warning: forcing requested degeneracies by using mean of hybridizations')
                    # to really have degenerate bands, we now use the mean values as input:
                    self.par.vBath[self.par.degeneracy[iD][:],:] = meanVals
                    meanVals = np.mean(par.epsImp[par.degeneracy[iD]])
                    self.par.epsImp[self.par.degeneracy[iD][:]] = meanVals
                    self.par.epsImp[self.parEff.degeneracy[iD][:]] = meanVals
       
    def pointToVecOrbStitch(self,point,orb,restPoint):
        #%% 
        # stitch the point into the restPoint
        combinedPoint = restPoint.copy()
        if self.par.numDeg == 0:
            lengthPoint = combinedPoint.size // self.par.NimpOrbs
        else:
            lengthPoint = combinedPoint.size // self.par.numDeg
        combinedPoint[orb*lengthPoint : (orb+1)*lengthPoint] = point
        
        #calculate the coeffs for the restPoint
        coeffD, coeffC = self.pointToVec(self.cfg,self.par,combinedPoint)
        
        return coeffD, coeffC    
    
    def calcEnergyHFFixEDOrb(self,pointX,*args):
    #%% 
        self.pointBasis = pointX
        #par = args[0]
        #hFSol = args[1]
        #edSol = args[2]
        orb = args[0]
        restPoint= args[1]
        #cfg = args[5]
        
        #start = time.time()
        if self.par.NimpOrbs > 5:
            print('deriv start',end=" ")
        self.cD, self.cC = self.pointToVecOrbStitch(pointX,orb,restPoint)
        #print('pointX')
        #print(pointX)
        
        #print np.dot(coeffsC1[0,:],coeffsC1[0,:]), np.dot(coeffsD[0,:],coeffsC1[0,:]), np.dot(coeffsD[0,:],coeffsD[0,:]),
        #self.cC = self.cC / np.sqrt(np.sum(self.cC[:,:]**2,axis=1))[:,np.newaxis]
        #self.cD = self.cD / np.sqrt(np.sum(self.cD[:,:]**2,axis=1))[:,np.newaxis]
        #for iO in range(self.par.NimpOrbs):
        #    self.cC[iO,:] /= np.sqrt(np.sum(self.cC[iO,:]**2))
        #    self.cD[iO,:] /= np.sqrt(np.sum(self.cD[iO,:]**2))
        #print('constrains')
        #print(np.dot(coeffsC1[0,:],coeffsC1[0,:].transpose()))
        #print(np.dot(coeffsD[0,:],coeffsD[0,:].transpose()))
        #print(np.dot(coeffsC1[0,:],coeffsD[0,:].transpose()))
        #print('startstuff',(time.time() - start)*1000.0)
        # set up uncorrelated rest Problem:
        #start = time.time()
        onePartHamilUnCorrSubOrb, coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(self.cfg,self.par,self.localHamiltonian,self.cD,self.cC)
        #np.savetxt('matAll.dat',onePartHamilUnCorrSub)
        #for iO in range(3):
        #    np.savetxt('mat'+str(iO)+'.dat',onePartHamilUnCorrSubOrb[iO,:,:])
        #print('setup',(time.time() - start)*1000.0)
    
        # solve the uncorrelated Problem blockwise:
        #start = time.time()
        unCorrDenMatOrb, PhiUnCorr, energyEffUnCorr  = hel.solveUncorrelatedProblem(self.par,onePartHamilUnCorrSubOrb)
        #print('solving',(time.time() - start)*1000.0)
        #start = time.time()
        _,_,energy, self.completeDenMatLocBasis,twoPartLocUpDn, twoPartLocUpUp = hel.stitchEnergy(self.cfg, self.par, self.edSol['corrDenMat'],self.edSol['twoPart'],unCorrDenMatOrb,coeffsComplete,coeffsOrbComplete)
        #print('stiching',(time.time() - start)*1000.0)
        # <H - H*>*
        
        # updating self constistent local HF hamiltonian
        #if self.cfg['hf']['updateUncorrelated']:
        #    hFSol = harFock.updateHartreeFock(self.par,self.cfg,self.completeDenMatLocBasis[:self.par.NimpOrbs,:self.par.NimpOrbs])
        #    self.localHamiltonian = hFSol['hamilLoc']

        
        energyEff = energyEffUnCorr + self.edSol['energyEffCorr']
        PhiComplete = self.edSol['PhiCorr'] + PhiUnCorr
    
        diffEnergy = energy - energyEff
        
        phiTilde = PhiComplete + diffEnergy
        #print(phiTilde)
    #    if len(pointX)>20:
    #        exit()
        return phiTilde
    def varVectorToParams(self):
        #%% 
        self.checkFirstGuess()
        print('this is the vector')
        print(self.point)
        if self.cfg['algo']['optU']:
            if self.parEff.NimpOrbs == 1:
                self.parEff.UImp = self.point[0]
                # cut away first entry
                restPoint = self.point[1:]
                
            else:
                self.parEff.UImp = self.point[0]
                self.parEff.JImp = self.point[1]
                # cut away first two entries
                restPoint = self.point[2:] 
            self.parEff.uMatrix = uMat.uMatrixWrapper(self.parEff,self.cfg)
        else:
            # pass the vector to next if clause
            restPoint = self.point[:]
        
        if self.cfg['algo']['optEpsD']:
            epsTilde = np.zeros(shape=(self.par.NimpOrbs,1))
            for iD in range(self.par.numDeg):
                for iOD in range(len(self.par.degeneracy[iD])):
                    epsTilde[self.par.degeneracy[iD][iOD],0] = restPoint[iD]
            self.parEff.epsImp = epsTilde[:,0]
            # cut away first numDeg entries
            restPoint = restPoint[self.par.numDeg:]

            
        if self.cfg['algo']['optV']:
            vTilde = np.zeros(shape=(self.par.NimpOrbs,1))
            for iD in range(self.par.numDeg):
                for iOD in range(len(self.par.degeneracy[iD])):
                    vTilde[self.par.degeneracy[iD][iOD],0] = restPoint[iD]
            self.parEff.vBath[:,0] = vTilde[:,0]
            # cut away first numDeg entries
            restPoint = restPoint[self.par.numDeg:]

            
        if self.cfg['algo']['optEpsBath']:
            epsBath = np.zeros(shape=(self.par.NimpOrbs,1))
            for iD in range(self.par.numDeg):
                for iOD in range(len(self.par.degeneracy[iD])):
                    epsBath[self.par.degeneracy[iD][iOD],0] = restPoint[iD]
            self.parEff.epsBath[:,0] = epsBath[:,0]
        #cfg['algo']['optOnlyBath'] = True
        
    def paramsToVarVector(self):
        #%% 
        self.checkFirstGuess()
        print('this is the vector')
        print(self.point)
        if self.cfg['algo']['optU']:
            if self.parEff.NimpOrbs == 1:
                self.parEff.UImp = self.point[0]
                # cut away first entry
                restPoint = self.point[1:]
                
            else:
                self.parEff.UImp = self.point[0]
                self.parEff.JImp = self.point[1]
                # cut away first two entries
                restPoint = self.point[2:] 
            self.parEff.uMatrix = uMat.uMatrixWrapper(self.parEff,self.cfg)
        else:
            # pass the vector to next if clause
            restPoint = self.point[:]
        
        if self.cfg['algo']['optEpsD']:
            epsTilde = np.zeros(shape=(self.par.NimpOrbs,1))
            for iD in range(self.par.numDeg):
                for iOD in range(len(self.par.degeneracy[iD])):
                    epsTilde[self.par.degeneracy[iD][iOD],0] = restPoint[iD]
            self.parEff.epsImp = epsTilde[:,0]
            # cut away first numDeg entries
            restPoint = restPoint[self.par.numDeg:]

            
        if self.cfg['algo']['optV']:
            vTilde = np.zeros(shape=(self.par.NimpOrbs,1))
            for iD in range(self.par.numDeg):
                for iOD in range(len(self.par.degeneracy[iD])):
                    vTilde[self.par.degeneracy[iD][iOD],0] = restPoint[iD]
            self.parEff.vBath[:,0] = vTilde[:,0]
            # cut away first numDeg entries
            restPoint = restPoint[self.par.numDeg:]

            
        if self.cfg['algo']['optEpsBath']:
            epsBath = np.zeros(shape=(self.par.NimpOrbs,1))
            for iD in range(self.par.numDeg):
                for iOD in range(len(self.par.degeneracy[iD])):
                    epsBath[self.par.degeneracy[iD][iOD],0] = restPoint[iD]
            self.parEff.epsBath[:,0] = epsBath[:,0]
        #cfg['algo']['optOnlyBath'] = True
        
        
    
    def calcPhiTildeED(self,x):
        #%% 

        self.point = x
        
        if self.par.NimpOrbs > 5:
            print('iter start')
            print()
    
        # put entries of the point x into parEff
        self.varVectorToParams()  
                    
        if (self.pointBasis is None) == False:
            pass
        else:
            #print('making new')
            self.cD, self.cC = harFock.firstGuessHFSubSpace(self.par)
            self.pointBasis =hel.vecToPoint(self.cfg, self.par, self.cD, self.cC)
            
            # if no point is present, use original ls -terms
        start = time.time()
        print('solving ED:')
        self.edSol = hel.funcEDPhi(self.cfg,self.par,self.parEff,self.blockCar,self.ham)
    
        print('ED finished after '+str(time.time()-start)+ ' sec')
        #print 'warning hacked code'
        #startCoeffsD, startCoeffsC1 = harFock.firstGuessHFSubSpace(par)
        #point = vecToPoint(par, startCoeffsD, startCoeffsC1)
        gOrbOrtho = dict()
        gOrbNormC = dict()
        gOrbNormD = dict()
    
        currentFixPoint = copy.copy(self.pointBasis)
        if self.par.numDeg == 0:
            lengthPoint = currentFixPoint.size//self.par.NimpOrbs
        else:
            lengthPoint = currentFixPoint.size//self.par.numDeg
    
        start = time.time()
    

        for iD in range(self.par.numDeg):
            if self.cfg['algo']['optOnlyBath']:
                gOrbNormC = lambda x: np.dot(self.pointToVecOrbStitch(x,iD,currentFixPoint)[1][self.par.degeneracy[iD][0],:],self.pointToVecOrbStitch(x,iD,currentFixPoint)[1][self.par.degeneracy[iD][0],:].transpose()) - 1.0
                consOrb = ({'type' : 'eq', 'fun' : gOrbNormC})

            else:
                #gOrbOrtho = lambda x: np.dot(self.pointToVecOrbStitch(x,iD,currentFixPoint)[0][self.par.degeneracy[iD][0],:],self.pointToVecOrbStitch(x,iD,currentFixPoint)[1][self.par.degeneracy[iD][0],:].transpose()) - 0.0
                gOrbNormC = lambda x: np.dot(self.pointToVecOrbStitch(x,iD,currentFixPoint)[1][self.par.degeneracy[iD][0],:],self.pointToVecOrbStitch(x,iD,currentFixPoint)[1][self.par.degeneracy[iD][0],:].transpose()) - 1.0
                gOrbNormD = lambda x: np.dot(self.pointToVecOrbStitch(x,iD,currentFixPoint)[0][self.par.degeneracy[iD][0],:],self.pointToVecOrbStitch(x,iD,currentFixPoint)[0][self.par.degeneracy[iD][0],:].transpose()) - 1.0
                consOrb = ({'type' : 'eq', 'fun' : gOrbOrtho},
                           {'type' : 'eq', 'fun' : gOrbNormC},
                           {'type' : 'eq', 'fun' : gOrbNormD})
                consOrb = ({'type' : 'eq', 'fun' : gOrbNormC},
                           {'type' : 'eq', 'fun' : gOrbNormD})
                #print(consOrb)
                #print('constrains')
                #for ii in range(3):
                #    print(consOrb[ii]['fun'](currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint]))
            #xMin,fMin,its,eMode,sMode = optimize.fmin_slsqp(calcEnergyHFFixEDOrb,currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint],listCons,args=(par,hFSol,edSol,iD,currentFixPoint))
            #_minimize_slsqp(fun, x0, args, jac, bounds,
            #                constraints, callback=callback, **options)
            #currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint] = copy.copy(xMin)
            #print('constrain optim for orb ' + str(iD)+' '+str(time.time() - start)+'s after',its,'iterations', fMin, pointX)
            o = optimize.minimize(self.calcEnergyHFFixEDOrb, x0=currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint], args=(iD,currentFixPoint), constraints=consOrb, method='SLSQP', options={'disp': False,'ftol' : self.cfg['algo']['deltaInner'],'maxiter':self.cfg['algo']['maxIterInner']})
            print('success',o.success,'with',o.message)
            
            currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint] = copy.copy(o.x)
            print('constrain optim for orb {0:d} in {1:1.2f}s after {2:d} funcs, {3:d} iterations:'.format(iD,time.time() - start,o.nfev,o.nit), o.fun)
            
    
       
        self.innerSuccess = o.success           
        print('one particle optimization finished with PhiTilde=',o.fun,self.point)
        #self.cD, self.cC = hel.pointToVec(self.cfg,self.par,currentFixPoint)
        
        # update HF solution
        if self.cfg['hf']['updateUncorrelated']:
            hFSol = harFock.updateHartreeFock(self.par,self.cfg,self.completeDenMatLocBasis[:self.par.NimpOrbs,:self.par.NimpOrbs])
            self.localHamiltonian = hFSol['hamilLoc']
            
        return o.fun
    
    
    def opt(self):
    #def opt(cfg,point,ham,par,parEff,blockCar,hfSol):
        #%%     
        #checkFirstGuess(point,par)
        self.outerSuccess = False
        self.checkFirstGuess()
        self.parEff.twoPart = hel.set_ham_pure_twoPart(self.parEff,self.ham)
        print('implement first guess checker!')
        if self.cfg['algo']['optOnlyBath']:
            print('optimizing only bath basis')
        else:
            print('optimizing complete basis')
        bfgsOpts = {'gtol' : self.cfg['algo']['deltaOuter']}
        NMOpts = {'xtol' : self.cfg['algo']['deltaOuter'],
                  'ftol' : self.cfg['algo']['deltaOuter'],
                  'maxfev': self.cfg['algo']['maxIterOuter']}
        powOpts = {'ftol': self.cfg['algo']['deltaOuter']}
    
        self.out=dict()
        # calculate two particle part of hamiltonian, this is unchanged in any case
        
        
        # optimizing    
        if self.cfg['algo']['maxIterOuter'] == 0:
            print('performing no outer loop! (no optimization of parameters of Hamiltonian)')
            self.fopt = self.calcPhiTildeED(self.point)
            self.func_calls = 0
            
            print('optimal PhiTilde', self.fopt,'after only optimizing basis',self.point)
        else:
            print("optimizing parameters of Hamiltonian (",len(self.point),"parameters)...",end=" ")
            o=optimize.minimize(self.calcPhiTildeED,self.point,method='Nelder-Mead',options=NMOpts)
            #o=optimize.minimize(self.calcPhiTildeEDWOCons,self.point,method='Powell')
            
            #o = optimize.basinhopping(self.calcPhiTildeEDWOCons,self.point)
            #,minimizer_kwargs={method:'Nelder-Mead',options:NMOpts}
            #o=optimize.minimize(calcPhiTildeED,point,args=(ham,par,parEff,spPars,blockCar,hfSol),method='BFGS',options=bfgsOpts)
            self.func_callsBFGS = o.nfev
            o.success = True
            if o.success == False:
                print('BFGS did not succeed. Using Nelder-Mead...')
                o=optimize.minimize(self.calcPhiTildeED,self.point,method='BFGS',options=NMOpts)
                self.func_callsNM = o.nfev
            else:
                self.func_callsNM = 0
            self.outerSuccess = o.success
            self.point = o.x
            self.fopt = o.fun
            self.func_calls = self.func_callsNM + self.func_callsBFGS
            print('optimal PhiTilde', self.fopt, 'number of BFGS steps:', self.func_callsBFGS,'NM steps:', self.func_callsNM,self.point)
    
    
    def calcUpdatedObservables(self):
    #%% 

        onePartHamilUnCorrSubOrb ,coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(self.cfg,self.par,self.localHamiltonian,self.cD,self.cC)
        
        # solve the uncorrelated Problem orbitlly blockwise:
        unCorrDenMatOrb,PhiUnCorr, energyEffUnCorr = hel.solveUncorrelatedProblem(self.par,onePartHamilUnCorrSubOrb)
    
        energyOne,energyTwo,energy, completeDenMatLocBasis,twoPartLocUpDn, twoPartLocUpUp = hel.stitchEnergy(self.cfg, self.par,self.edSol['corrDenMat'],self.edSol['twoPart'],unCorrDenMatOrb,coeffsComplete,coeffsOrbComplete)
        energyEff = energyEffUnCorr + self.edSol['energyEffCorr']
        PhiComplete = self.edSol['PhiCorr'] + PhiUnCorr
        print('\nDetailed energy and free energy')
        print('phi* C:      {:+1.6f}'.format(self.edSol['PhiCorr']))
        print('phi* C\':     {:+1.6f}'.format(PhiUnCorr))
        print('phi* C+C\':   {:+1.6f}'.format( PhiComplete))
        print('<H*>* C:     {:+1.6f}'.format(self.edSol['energyEffCorr']))
        print('<H*>* C\':    {:+1.6f}'.format(energyEffUnCorr))
        print('<H*>* C+C\':  {:+1.6f}'.format( energyEff))
        print('<H>*:        {:+1.6f}'.format(energy))
        print('<H_one>*:        {:+1.6f}'.format(energyOne))
        print('<H_two>*:        {:+1.6f}'.format(energyTwo))
        print('~Phi = phi* + <H>* - <H*>*:',PhiComplete+energy-energyEff)
        print('')
        sSquare = 0.0
        sSquareB = 0.0
        # calculate <S^2> =< 0.5* (S+S- + S-S+) + Sz^2>
        # <Sz^2>
        # factor 2 from spin degeneracy, factor 1/2*1/2 from sz eigenvalue
        #print('edSolCorrDenMat')
        #print(edSol['corrDenMat'])
        #print('corrDen')
        #print(completeDenMatLocBasis) 
        SzSquare = 0.0
        for iO in range(self.par.NimpOrbs):
            SzSquare += 0.5*completeDenMatLocBasis[iO,iO]
            for jO in range(self.par.NimpOrbs):
                SzSquare += -0.5* twoPartLocUpUp[iO,jO,iO,jO]- 0.5*twoPartLocUpDn[iO,jO,jO,iO]
        # <S+S- + S-S+>
        SpmmpSquare = 2.0 * np.trace(completeDenMatLocBasis[:self.par.NimpOrbs,:self.par.NimpOrbs])
        for iO in range(self.par.NimpOrbs):
            for jO in range(self.par.NimpOrbs):
                SpmmpSquare += -2.0 * twoPartLocUpDn[iO,jO,iO,jO]
    
        sSquare = SzSquare + 0.5*SpmmpSquare
        
        #for ii in range(par.NimpOrbs):
        #    # factor 0.5 from definition, factor 4 from calculus and
        #    # spin-degeneracy, factor /2.0 from totalLocDenMat is multiplied with 2 allready
        #    sSquare += 0.5*4.0*(completeDenMatLocBasis[ii,ii]/2.0)
        #    sSquareB += 0.5*4.0*(completeDenMatLocBasis[ii,ii]/2.0)
        #    for jj in range(par.NimpOrbs):
        #        sSquare += 0.5*( -2.0 * twoPartLocUpDn[ii,jj,ii,jj] -2.0 * twoPartLocUpUp[ii,jj,ii,jj]- 2.0 * twoPartLocUpDn[ii,jj,jj,ii] )
        #        sSquareB += 0.5*( -4.0 * twoPartLocUpDn[ii,jj,ii,jj] - 2.0 * twoPartLocUpUp[ii,jj,jj,ii] )
                
        print('sSquare from easyFour', sSquare)
        #print 'sSquare from easyFour', sSquareB
        sSquareB = 0.0
        return completeDenMatLocBasis, energy, twoPartLocUpUp, twoPartLocUpDn, coeffsComplete, sSquare, sSquareB

        

