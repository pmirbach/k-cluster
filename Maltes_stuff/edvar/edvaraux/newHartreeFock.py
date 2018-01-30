# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:08:27 2014

@author: mschueler
"""

import numpy as np
import sys


#import matplotlib.pyplot as plt
from scipy import optimize
import time

import copy

def fermiFunc(par,energies):
    #%% 
    if par.onlyGroundState:
        fermiFunction =  (-np.sign(energies - par.mu)+1)*0.5
        fermiFunction[np.abs(energies-par.mu)<1e-15] = 0.5
    else:
        # this leads to overflows for small betas
        #fermiFunction = 1.0 / ( np.exp((energies-par.mu)*par.beta) + 1 ) 
        # this does not lead to overflow:
        fermiFunction = 0.5* (1.0 - np.tanh((energies-par.mu)*par.beta*0.5))
    return fermiFunction
    
    
def onePartMatrix(epsk,Vk,epsdMatrix,nBath):
    #%% 
    nOrbs = Vk.shape[0]
    hamUp = np.zeros(shape=(nOrbs+nBath*nOrbs,nOrbs+nBath*nOrbs))
    hamUp[:nOrbs,:nOrbs] = epsdMatrix[:nOrbs,:nOrbs]
    hamUp[nOrbs:,nOrbs:][np.diag_indices(nBath*nOrbs)] = np.reshape(epsk,epsk.size)
    for iO in range(nOrbs):

        fromInd = nOrbs+iO*nBath
        toInd = nOrbs+(iO+1)*nBath
        hamUp[iO,fromInd:toInd] = Vk[iO,:]
        hamUp[fromInd:toInd,iO] = Vk[iO,:]

    return hamUp
    
def onePartMatrixOrb(epsk,Vk,epsdMatrix,nBath):
    #%% 
    nOrbs = Vk.shape[0]
    hamUp = np.zeros(shape=(nOrbs,nBath+1,nBath+1))
    for iO in range(nOrbs):
        hamUp[iO,0,0] = epsdMatrix[iO,iO]
        for iB in range(nBath):
            hamUp[iO,1+iB,1+iB] = epsk[iO,iB]
            hamUp[iO,0,1+iB] = Vk[iO,iB]
            hamUp[iO,1+iB,0] = Vk[iO,iB]
            
    return hamUp
    
def onePartMatrixAllSpin(epsk,Vk,epsdMatrix,states):
    #%% 
    # everything is spin dependent
    # this is more like Indra format
    nOrbs = epsdMatrix.shape[0]
    nBath = epsk.shape[0]
    
    hamUp = np.zeros(shape=(nOrbs+nBath,nOrbs+nBath),dtype=complex)
    hamUp[:nOrbs,:nOrbs] = epsdMatrix[:nOrbs,:nOrbs]

    hamUp[nOrbs:,nOrbs:] = epsk
    for iB in range(nBath):
        # spin up 
        #bath energies
        Ind = nOrbs+iB

        # hybridization
        hamUp[:nOrbs,Ind] = Vk[iB]*states[iB,:]
        hamUp[Ind,:nOrbs] = Vk[iB]*states[iB,:]


    return hamUp
def densityMatrixCorr(par,eigVals,eigVecs):
    #%% 
    fermi = fermiFunc(par,eigVals)
    occMat = np.diag(fermi)
    denMat = (np.dot(np.dot((eigVecs),occMat),np.transpose(eigVecs)))[:par.NimpOrbs,:par.NimpOrbs]

    return denMat

def densityMatrix(par,eigVals,eigVecs):
    #%% 
    fermi = fermiFunc(par,eigVals)
    occMat = np.diag(fermi)
    denMat = (np.dot(np.dot((eigVecs),occMat),np.transpose(eigVecs)))
    return denMat
    
def densityMatrixWOParCan( nElec, eigVals,eigVecs):
    #%% 
    up = eigVals[nElec-1]
    degenerate = np.abs(eigVals - up) < 1e-14
    num = eigVals[degenerate].size
    fermi = np.zeros(eigVals.shape)
    fermi[:nElec] = 1.0
    # now we have the right electron number
    # but for the case of degeneracies we have to be carfull
    # case 1) all degenerate states are filled, then we are good
    # case 2) not all degenerate states are filled, then we put 
    #         1/numberOfDeg into all degenerate states
    if np.all(fermi[degenerate] == 1.0):
        pass
        #print 'we are good'
    else:
        fermi[degenerate] = (nElec-np.sum(fermi[degenerate == False]))/float(num)

   
    occMat = np.diag(fermi)
    denMat = (np.dot(np.dot((eigVecs),occMat),np.conjugate(np.transpose(eigVecs)))).real
    return denMat    
def selfEnergy(uMatrix,densityMatrix):
    #%% 
    nOrbs = uMatrix.shape[0]
    sigma = np.zeros(shape=(nOrbs,nOrbs))
    #sigmaBeta = np.zeros(shape=(nOrbs,nOrbs))
    for ii in range(nOrbs):
        for jj in range(nOrbs):
            sigma[ii,jj] = 0.5*(2.0*np.sum(uMatrix[:,ii,:,jj]*densityMatrix[:nOrbs,:nOrbs]) + 2.0*np.sum(uMatrix[ii,:,jj,:]*densityMatrix[:nOrbs,:nOrbs])
                               -1.0*np.sum(uMatrix[ii,:,:,jj]*densityMatrix[:nOrbs,:nOrbs]) - 1.0*np.sum(uMatrix[:,ii,jj,:]*densityMatrix[:nOrbs,:nOrbs]))
    return sigma

def selfEnergyLS(uMatrix,densityMatrix):
    
    sigma = np.zeros(shape=densityMatrix.shape)
    # spin up up
    
    dMUpUp = densityMatrix[:5,:5]
    dMDnDn = densityMatrix[5:10,5:10]
    dMUpDn = densityMatrix[:5,5:10]
    dMDnUp = densityMatrix[5:10,:5]
    for ii in range(5):
        for jj in range(5):
            a1 = np.sum(uMatrix[ii,:,jj,:]*(dMUpUp + dMDnDn))
            a2 = np.sum(uMatrix[:,ii,:,jj]*(dMUpUp + dMDnDn))
            # up up
            # hartree
            sigma[ii,jj] += a1 + a2
            # fock            
            sigma[ii,jj] -= np.sum(uMatrix[ii,:,:,jj]*dMUpUp)
            sigma[ii,jj] -= np.sum(uMatrix[:,ii,jj,:]*dMUpUp)
            
            # dn dn
            # hartree
            sigma[ii+5,jj+5] += a1 + a2
            # fock
            sigma[ii+5,jj+5] -= np.sum(uMatrix[ii,:,:,jj]*dMDnDn)
            sigma[ii+5,jj+5] -= np.sum(uMatrix[:,ii,jj,:]*dMDnDn)
            # up dn
            # fock
            sigma[ii,jj+5] -= np.sum(uMatrix[ii,:,:,jj]*dMDnUp)
            sigma[ii,jj+5] -= np.sum(uMatrix[:,ii,jj,:]*dMDnUp)
            # dn up
            # fock
            sigma[ii+5,jj] -= np.sum(uMatrix[ii,:,:,jj]*dMUpDn)
            sigma[ii+5,jj] -= np.sum(uMatrix[:,ii,jj,:]*dMUpDn)
    # double counting 1/2
    sigma = 0.5*sigma
            #for kk in range(5):
            #    for ll in range(5):
                    #a1 = uMatrix[ii,kk,jj,ll]*(densityMatrix[kk,ll]
                    #                                +densityMatrix[kk+5,ll+5])
                    #a2 = uMatrix[kk,ii,ll,jj]*(densityMatrix[kk,ll]
                    #                                +densityMatrix[kk+5,ll+5])
                    ## hartree up up
                    #sigma[ii,jj] += a1 + a2
                    ## hartree dn dn 
                    #sigma[ii+5,jj+5] += a1 + a2
                    #
                    #sigma[ii,jj] -= uMatrix[ii,kk,ll,jj]*densityMatrix[kk,ll]
                    #sigma[ii,jj] -= uMatrix[kk,ii,jj,ll]*densityMatrix[kk,ll]
                    #
                    #sigma[ii+5,jj+5] -= uMatrix[ii,kk,ll,jj]*densityMatrix[kk+5,ll+5]
                    #sigma[ii+5,jj+5] -= uMatrix[kk,ii,jj,ll]*densityMatrix[kk+5,ll+5]
                    
                    # spin up dn

                    #sigma[ii,jj+5] -= uMatrix[ii,kk,ll,jj]*densityMatrix[kk+5,ll]
                    #sigma[ii,jj+5] -= uMatrix[kk,ii,jj,ll]*densityMatrix[kk+5,ll]
                    
                    # spin dn up

                    #sigma[ii+5,jj] -= uMatrix[ii,kk,ll,jj]*densityMatrix[kk,ll+5]
                    #sigma[ii+5,jj] -= uMatrix[kk,ii,jj,ll]*densityMatrix[kk,ll+5]
                    

        
    
    return sigma    
def selfEnergyUnres(uMatrix,dmP,dmPS):
    #%% 
    nOrbs = uMatrix.shape[0]
    sigma = np.zeros(shape=(nOrbs,nOrbs))
    #sigmaBeta = np.zeros(shape=(nOrbs,nOrbs))
    for ii in range(nOrbs):
        for jj in range(nOrbs):
            sigma[ii,jj] = 0.5*(np.sum(uMatrix[:,ii,:,jj]*(dmP+dmPS)) + np.sum(uMatrix[ii,:,jj,:]*(dmP+dmPS))
                               -np.sum(uMatrix[ii,:,:,jj]*dmP) - np.sum(uMatrix[:,ii,jj,:]*dmP))
    return sigma    

    
def solveHartreeFockUnres(par,cfg):
    #%%  
    hFSol = dict()
    if cfg['algo']['noHF']:
        pass
    else:
        print('solving unrestricted HF-problem')
        epsdMatrix=  np.diag(par.epsImp)
    
        hamNonInt=onePartMatrix(par.epsBath,par.vBath,epsdMatrix,par.Nbath)
    
        eigValsFirst,eigVecsFirst= np.linalg.eigh(hamNonInt,UPLO='U')
        denMat = [None,None]
        oldOcc = [None,None]
        sigma= [None,None]
        oldSigma = [None,None]
        ham = [None,None]
        eigVals= [None,None]
        eigVecs = [None,None]
        
        for iS in range(2):
            denMat[iS] = densityMatrixCorr(par,eigValsFirst,eigVecsFirst)
        
        if par.NimpOrbs > 2:
            for iS in range(2):
                denMat[iS][0,0] = 0.75744756
                denMat[iS][1,1] = 0.75744756
                denMat[iS][2,2] = 0.84603203
                denMat[iS][3,3] = 0.75744756
                denMat[iS][4,4] = 0.84603203
        else:
            denMat[0][0,0] = 1.0
            denMat[1][0,0] = 0.0
        #denMat[0][0,0] = 0.5
        #denMat[1][0,0] = 0.0
        #denMat = 0.5*np.eye(par.NimpOrbs)
        #denMat = 0.0589*np.eye(nOrbs)
        #sigUp = selfEnergy(par.uMatrix,denMatUp)
        #sigDn = selfEnergy(par.uMatrix,denMatUp)
        for iS in range(2):
            sigma[iS] = selfEnergyUnres(par.uMatrix,denMat[(iS)%2],denMat[(iS+1)%2])#+(iS-0.5)*50.0

            
        for ii in range(cfg['hf']['maxIter']):
            for iS in range(2):
                oldOcc[iS] = np.diag(denMat[iS][:par.NimpOrbs,:par.NimpOrbs])
                        
            oldSigma = copy.copy(sigma)
            
            for iS in range(2):
                ham[iS] = onePartMatrix(par.epsBath,par.vBath,epsdMatrix+sigma[iS] ,par.Nbath)
                eigVals[iS],eigVecs[iS]=np.linalg.eigh(ham[iS],UPLO='U')
                denMat[iS] = densityMatrixCorr(par,eigVals[iS],eigVecs[iS])

            diff = np.max(np.abs(np.diag(denMat[0][:par.NimpOrbs,:par.NimpOrbs])- oldOcc[0]))
            
            for iS in range(2):
                if ii < 10:
                    mag =0.0# +(iS-0.5)*1.0/float(ii+1)
                else:
                    mag = 0.0
                sigma[iS] = selfEnergyUnres(par.uMatrix,denMat[(iS)%2],denMat[(iS+1)%2])+(iS-0.5)*1.0/float(ii+1)+mag
                
            for iS in range(2):
                sigma[iS] = sigma[iS]*(1-cfg['hf']['mixing']) + oldSigma[iS]*cfg['hf']['mixing']
            
            
        
            print('HF iteration {0:d} diff {1:1.2g} n_d_0 {2:1.4g} {3:1.4g} {4:1.4g}'.format(ii,diff,np.trace(denMat[0]),np.trace(denMat[1]),mag))
            if diff < cfg['hf']['minDiff']:
                print('desired accuracy reached')
                break
            if ii == cfg['hf']['maxIter']-1:
                print('HF not converged yet!')
        hFSol['eigVals']=eigVals
        hFSol['eigVecs']=eigVecs
        
        if True:
            fermi = [None,None]
            denMatFull = [None,None]
            twoPart = [[None,None],[None,None]]
        # calculate PhiTilde = Phi + <H-H*>*
            for iS in range(2):
                fermi[iS] = fermiFunc(par,eigVals[iS])
        # Phi
                denMatFull[iS] = densityMatrix(par,eigVals[iS],eigVecs[iS])


            if par.onlyGroundState:
                Phi = 0.0
                for iS in range(2):
                    Phi += np.sum(eigVals[iS] *fermi[iS])
            else:
                eigValsAll = np.hstack((eigVals[0],eigVals[1]))

                Phi = -1.0/par.beta * np.sum(np.log(1+np.exp(-par.beta * (eigValsAll - par.mu))))
                # if temperature is too small, this will likely result in inf
                # then simply use the groundstate energy as the free energy.
                if np.isinf(Phi):
                    Phi =  2*np.sum(eigValsAll *fermiFunc(par,eigValsAll))


        # <H*>*
            effEnergy = 0.0
            for iS in range(2):
                effEnergy += np.sum(eigVals[iS]*fermi[iS])

        # <H>*
        # first the one particle part:
            onePartEnergy = 0.0
            for iS in range(2):
                onePartEnergy += np.sum(denMatFull[iS] * hamNonInt)
        # second the two particle part:
        # calculate the two particle density matrix for the impurity block:
            for iS in range(2):
                for jS in range(2):
                    twoPart[iS][jS] = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))

                    for ii in range(par.NimpOrbs):
                        for jj in range(par.NimpOrbs):
                            for kk in range(par.NimpOrbs):
                                for ll in range(par.NimpOrbs):
                                    if iS == jS:
                                        twoPart[iS][jS][ii,jj,kk,ll] = (denMatFull[iS][ii,ll] * denMatFull[jS][jj,kk]
                                                                   -denMatFull[iS][ii,kk] * denMatFull[jS][jj,ll])
                                    else:
                                        twoPart[iS][jS][ii,jj,kk,ll] = (denMatFull[iS][ii,ll] * denMatFull[jS][jj,kk])
                                                                   

            twoPartEnergy = 0.0
        # factor two of u-matrix cancels because of the spin degeneracy:
        # DnDn and DnUp are not considered...
            for iS in range(2):
                for jS in range(2):
                    for ii in range(par.NimpOrbs):
                        for jj in range(par.NimpOrbs):
                            for kk in range(par.NimpOrbs):
                                for ll in range(par.NimpOrbs):
                                    twoPartEnergy += 0.5*par.uMatrix[ii,jj,ll,kk] * (
                                                             twoPart[iS][jS][ii,jj,kk,ll] )

            origEnergy = onePartEnergy + twoPartEnergy
            phiTilde = Phi + origEnergy - effEnergy
            hFSol['denMat']=denMatFull
            print('phiTilde from HF', phiTilde)
            print('occ', np.diag(denMat[0])+np.diag(denMat[1]),'=',np.trace(denMat[0])+np.trace(denMat[1]))
            hFSol['phiTilde'] = phiTilde

            # calculate S^2
            # factor 2 from spin degeneracy                                          
            SzSquare = np.trace(denMatFull[0][:par.NimpOrbs,:par.NimpOrbs])+np.trace(denMatFull[1][:par.NimpOrbs,:par.NimpOrbs])
            for iO in range(par.NimpOrbs):
                for jO in range(par.NimpOrbs):
                    SzSquare += -1.0 * twoPart[0][0][iO,jO,iO,jO]-1.0 * twoPart[1][1][iO,jO,iO,jO] - 1.0 * twoPart[0][1][iO,jO,jO,iO]- 1.0 * twoPart[1][0][iO,jO,jO,iO]
            # <S+S- + S-S+>
            SpmmpSquare = np.trace(denMatFull[0][:par.NimpOrbs,:par.NimpOrbs])+np.trace(denMatFull[1][:par.NimpOrbs,:par.NimpOrbs])
            for iO in range(par.NimpOrbs):
                for jO in range(par.NimpOrbs):
                    SpmmpSquare += -2.0 * twoPart[0][1][iO,jO,iO,jO]
            sSquare = SzSquare + 0.5*SpmmpSquare
            print('sSquare from easyFour', sSquare)
            print('S',-0.5+np.sqrt(0.25+sSquare))
                                     
        

    return hFSol
 
def calcEnergy(mix,*args):
    #%% 
    par = args[0]
    SigVec = args[1]
    epsdMatrix=  np.diag(par.epsImp)
    
    sig = np.sum(SigVec*mix[np.newaxis,np.newaxis,:],2)
    ham = onePartMatrix(par.epsBath,par.vBath,epsdMatrix+sig ,par.Nbath)
    eigVals,eigVecs= np.linalg.eigh(ham,UPLO='U')
    fermi = fermiFunc(par,eigVals)
    occMat = np.diag(fermi)
    denMat = (np.dot(np.dot((eigVecs),occMat),np.transpose(eigVecs)))
    energy = np.sum(fermi*eigVals) + np.sum(sig*denMat[:par.NimpOrbs,:par.NimpOrbs])
    return energy
    
  
def updateHartreeFock(par,cfg,densMat):
    #%%  

    epsdMatrix=  np.diag(par.epsImp)
    
    ham = onePartMatrix(par.epsBath,par.vBath,epsdMatrix+selfEnergy(par.uMatrix,densMat) ,par.Nbath)
        
        
    eigVals,eigVecs= np.linalg.eigh(ham,UPLO='U')
        
    denMat = densityMatrixCorr(par,eigVals,eigVecs)         
                                              
    #print ham
    hFSol = dict()
    hFSol['eigVals'] = eigVals
    hFSol['eigVecs'] = eigVecs
    hFSol['denMat'] = denMat
    hFSol['hamilLoc'] = ham

        #hFSol['phiTilde'] = phiTilde
        

    return hFSol  


def solveHartreeFockDIIS(par,cfg):
    #%%  
    hFSol = dict()
    if cfg['algo']['noHF']:
        pass
    else:
        print('solving HF-problem')
        epsdMatrix=  np.diag(par.epsImp)
    
        hamNonInt=onePartMatrix(par.epsBath,par.vBath,epsdMatrix,par.Nbath)
    
        eigValsFirst,eigVecs= np.linalg.eigh(hamNonInt,UPLO='U')

        denMat = densityMatrixCorr(par,eigValsFirst,eigVecs)

        for iO in range(par.NimpOrbs):
            denMat[iO,iO] = 0.5

        
        sig = selfEnergy(par.uMatrix,denMat)
        
        nDIIS = 10
        matDIIS = np.zeros(shape=(nDIIS+1,nDIIS+1))
        matDIIS[-1,:-1] = -1
        matDIIS[:-1,-1] = -1
        solVec = np.zeros(shape=nDIIS+1)
        solVec[-1] = -1
        #diffVec = 100*np.ones(shape=(nDIIS,par.epsImp.size))
        pVec = np.zeros(shape=(nDIIS+1,hamNonInt.shape[0]))
        SigVec = np.ones(shape=(nDIIS,par.epsImp.size,par.epsImp.size))
        DenVec = np.ones(shape=(nDIIS,par.epsImp.size,par.epsImp.size))


        for ii in range(cfg['hf']['maxIter']):
            
            oldOcc = copy.copy(denMat)
            sig = selfEnergy(par.uMatrix,denMat)
            
            ham = onePartMatrix(par.epsBath,par.vBath,epsdMatrix+sig ,par.Nbath)
            
            eigVals,eigVecs= np.linalg.eigh(ham,UPLO='U')
            
            # update the solution vector p
            pVec[0,:] = np.sort(eigVals)
            
            # caluclate differences to prior iteration of solution vector
            deltaP = np.diff(pVec,axis=0)
            if False:
                print('history of eigenvalues')
                print(pVec)
                print('history of differences of e-values')
                print(deltaP)
            denMat = densityMatrixCorr(par,eigVals,eigVecs)
            
            diff = np.max(np.abs(np.diag(denMat[:par.NimpOrbs,:par.NimpOrbs])- oldOcc))
 
            # set up matrix:
            for jj in range(nDIIS):
                for kk in range(nDIIS):
                    matDIIS[jj,kk] = np.dot(deltaP[jj,:],deltaP[kk,:].transpose())
            if ii > nDIIS:
                mix = np.linalg.solve(matDIIS,solVec)[:-1]
            else:
                # linear mixing to start with
                mix = np.zeros(shape=nDIIS)
                mix[0] = 1-cfg['hf']['mixing']
                mix[1] = cfg['hf']['mixing'] 
       
            print('mix params')
            print(mix)
            # new self energy, unmixed
            SigVec[0,:,:] = copy.copy(selfEnergy(par.uMatrix,denMat))
            DenVec[0,:,:] = copy.copy(denMat)
            sig = np.zeros(sig.shape)
            denMat = np.zeros(denMat.shape)
            for iD in range(nDIIS):
                sig += SigVec[iD,:,:] * mix[iD]
                denMat += DenVec[iD,:,:] * mix[iD]
            #sig = np.sum(SigVec*mix[np.newaxis,np.newaxis,:],2)
            #sig = selfEnergy(par.uMatrix,denMat)*(1-parHF.mixing) +  parHF.mixing*oldSig
            #SigVec[:,:,0] = copy.copy(sig)
            #print 'old version', selfEnergy(par.uMatrix,denMat)*(1-parHF.mixing) +  parHF.mixing*oldSig
            #print 'new version', sig
        
            print('HF iteration',ii, 'diff', diff, 'n_d_0', np.diag(denMat)[0], np.trace(denMat))
            if diff < cfg['hf']['minDiff']:
                print('desired accuracy reached')
                break
            if ii == cfg['hf']['maxIter']-1:
                print('HF not converged yet!')
            pVec = np.roll(pVec,1,axis=0)
            #print(SigVec[:,0,0])
            SigVec = np.roll(SigVec,1,axis=0)
            DenVec = np.roll(DenVec,1,axis=0)
            #print(SigVec[0,0,:])
        
            #print sig[0,0]
            #print oldSig[0,0]
            
        if True:
        # calculate PhiTilde = Phi + <H-H*>*
            fermi = fermiFunc(par,eigVals)
        # Phi
            denMatFull = densityMatrix(par,eigVals,eigVecs)
            

            if par.onlyGroundState:
                Phi = 2.0*np.sum(eigVals *fermi)
            else:
                Phi = -2.0/par.beta * np.sum(np.log(1+np.exp(-par.beta * (eigVals - par.mu))))
                # if temperature is too small, this will likely result in inf
                # then simply use the groundstate energy as the free energy.
                if np.isinf(Phi):
                    Phi =  2*np.sum(eigVals *fermiFunc(par,eigVals))


        # <H*>*
            effEnergy = np.sum(eigVals*2.0*fermi)

        # <H>*
        # first the one particle part:
            onePartEnergy = 2.0*np.sum(denMatFull * hamNonInt)
        # second the two particle part:
        # calculate the two particle density matrix for the impurity block:
            twoPartUpUp = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
            twoPartUpDn = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
            for ii in range(par.NimpOrbs):
                for jj in range(par.NimpOrbs):
                    for kk in range(par.NimpOrbs):
                        for ll in range(par.NimpOrbs):
                            twoPartUpUp[ii,jj,kk,ll] = (denMatFull[ii,ll] * denMatFull[jj,kk]
                                                       -denMatFull[ii,kk] * denMatFull[jj,ll])
                            twoPartUpDn[ii,jj,kk,ll] = denMatFull[ii,ll] * denMatFull[jj,kk]
            twoPartEnergy = 0.0
        # factor two of u-matrix cancels because of the spin degeneracy:
        # DnDn and DnUp are not considered...
            for ii in range(par.NimpOrbs):
                for jj in range(par.NimpOrbs):
                    for kk in range(par.NimpOrbs):
                        for ll in range(par.NimpOrbs):
                            twoPartEnergy += par.uMatrix[ii,jj,ll,kk] * (
                                                     twoPartUpUp[ii,jj,kk,ll]
                                                    +twoPartUpDn[ii,jj,kk,ll]
                                                             )

            origEnergy = onePartEnergy + twoPartEnergy
            phiTilde = Phi + origEnergy - effEnergy
        
            print('phiTilde from HF', phiTilde)
            print('occ', np.diag(denMat))
            hFSol['phiTilde'] = phiTilde

            # calculate S^2
            # factor 2 from spin degeneracy                                          
            SzSquare = 2.0*np.trace(denMatFull[:par.NimpOrbs,:par.NimpOrbs])
            for iO in range(par.NimpOrbs):
                for jO in range(par.NimpOrbs):
                    SzSquare += -2.0 * twoPartUpUp[iO,jO,iO,jO] - 2.0 * twoPartUpDn[iO,jO,jO,iO]
            # <S+S- + S-S+>
            SpmmpSquare = 2.0 * np.trace(denMatFull[:par.NimpOrbs,:par.NimpOrbs])
            for iO in range(par.NimpOrbs):
                for jO in range(par.NimpOrbs):
                    SpmmpSquare += -2.0 * twoPartUpDn[iO,jO,iO,jO]
            sSquare = SzSquare + 0.5*SpmmpSquare
            print('sSquare from easyFour', sSquare)
                                     
                    
        
        #print ham
        #hFSol = dict()
        hFSol['eigVals'] = eigVals
        hFSol['eigVecs'] = eigVecs
        hFSol['denMat'] = denMatFull
        #hFSol['hamilLoc'] = np.dot(eigVecs,np.dot(np.diag(eigVals),eigVecs.transpose()))
        hFSol['hamilLoc'] = ham

        #hFSol['phiTilde'] = phiTilde
        

    return hFSol
    
    
def solveHartreeFock(par,cfg):
    #%%  
    hFSol = dict()
    if cfg['algo']['noHF']:
        pass
    else:
        print('solving HF-problem')
        epsdMatrix=  np.diag(par.epsImp)
    
        hamNonInt=onePartMatrix(par.epsBath,par.vBath,epsdMatrix,par.Nbath)
    
        eigValsFirst,eigVecs= np.linalg.eigh(hamNonInt,UPLO='U')

        denMat = densityMatrixCorr(par,eigValsFirst,eigVecs)

        if par.NimpOrbs > 4:
            denMat[0,0] = 0.75744756
            denMat[1,1] = 0.75744756
            denMat[2,2] = 0.84603203
            denMat[3,3] = 0.75744756
            denMat[4,4] = 0.84603203
        else:
            for iO in range(par.NimpOrbs):
                denMat[iO,iO] = 0.5
        #denMat = np.zeros((par.NimpOrbs,par.NimpOrbs))
        #denMat[0,0]=0.58392481
        #denMat[1,1]=0.58392481
        #denMat[2,2]=0.70924199
        #denMat[3,3]=0.58392481
        #denMat[4,4]=0.70924199
        #denMat = 0.5*np.eye(par.NimpOrbs)
        #denMat = 0.0589*np.eye(nOrbs)
        
        sig = selfEnergy(par.uMatrix,denMat)
        

        nDIIS = 2
        matDIIS = np.zeros(shape=(nDIIS+1,nDIIS+1))
        matDIIS[-1,:-1] = -1
        matDIIS[:-1,-1] = -1
        solVec = np.zeros(shape=nDIIS+1)
        solVec[-1] = -1
        diffVec = 100*np.ones(shape=nDIIS)
        SigVec = np.zeros(shape=(epsdMatrix.shape[0],epsdMatrix.shape[1],nDIIS))
        SigVec[:,:,1] = sig
        bounds = []
        for iD in range(nDIIS):
            bounds.append((0,1))
    

        for ii in range(cfg['hf']['maxIter']):
            print(np.diag(denMat))
            oldOcc = np.diag(denMat[:par.NimpOrbs,:par.NimpOrbs])
            #oldSig = copy.copy(sig)
            
            ham = onePartMatrix(par.epsBath,par.vBath,epsdMatrix+SigVec[:,:,1] ,par.Nbath)
            
            
            eigVals,eigVecs= np.linalg.eigh(ham,UPLO='U')
            
            denMat = densityMatrixCorr(par,eigVals,eigVecs)
            
            diff = np.max(np.abs(np.diag(denMat[:par.NimpOrbs,:par.NimpOrbs])- oldOcc))
            #dVec[0] = copy.copy()
            diffVec[0] = copy.copy(diff)
 
            # set up matrix:
            matDIIS[:-1,:-1] = np.outer(diffVec,diffVec)
            #con = lambda x : sum(x) - 1.0
            cons = {'type' : 'eq', 'fun' : lambda x : sum(x) - 1.0}
            if ii > 2000*nDIIS:
                #optimize.minimize()
                optMix = optimize.minimize(calcEnergy, x0 = np.ones(nDIIS)/np.float(nDIIS), args=(par,SigVec),method='SLSQP'
                                        , bounds=bounds, constraints=cons)
               #mix = np.linalg.solve(matDIIS,solVec)[:-1]
                print('DIIS mixing coefficients', optMix)
                mix = optMix.x
            else:
            
                mix = np.zeros(shape=nDIIS)
                mix[0] = 1-cfg['hf']['mixing']
                mix[1] = cfg['hf']['mixing']          

            #print 'init mixing coefficients', mix
        #print calcEnergy(mix,par,SigVec)
        
            SigVec[:,:,0] = selfEnergy(par.uMatrix,denMat)
            
            #print SigVec[:,:,0] * (1-parHF.mixing) + parHF.mixing     * SigVec[:,:,1]
            #print np.sum(SigVec * mix[np.newaxis,np.newaxis,:],axis=2)
            sig = np.zeros(sig.shape)
            for iD in range(nDIIS):
                sig += SigVec[:,:,iD] * mix[iD]
            sig = np.sum(SigVec*mix[np.newaxis,np.newaxis,:],2)
            #sig = selfEnergy(par.uMatrix,denMat)*(1-parHF.mixing) +  parHF.mixing*oldSig
            SigVec[:,:,0] = copy.copy(sig)
            #print 'old version', selfEnergy(par.uMatrix,denMat)*(1-parHF.mixing) +  parHF.mixing*oldSig
            #print 'new version', sig
        
            print('HF iteration {0:d} diff {1:1.2g} n_d_0 {2:1.4g}'.format(ii, diffVec[0],np.trace(denMat)))
            if diff < cfg['hf']['minDiff']:
                print('desired accuracy reached')
                break
            if ii == cfg['hf']['maxIter']-1:
                print('HF not converged yet!')
            diffVec = np.roll(diffVec,1)
            #print SigVec[0,0,:]
            SigVec = np.roll(SigVec,1,2)
            #print SigVec[0,0,:]
        
            #print sig[0,0]
            #print oldSig[0,0]
            
        if True:
        # calculate PhiTilde = Phi + <H-H*>*
            fermi = fermiFunc(par,eigVals)
        # Phi
            denMatFull = densityMatrix(par,eigVals,eigVecs)
            

            if par.onlyGroundState:
                Phi = 2.0*np.sum(eigVals *fermi)
            else:
                Phi = -2.0/par.beta * np.sum(np.log(1+np.exp(-par.beta * (eigVals - par.mu))))
                # if temperature is too small, this will likely result in inf
                # then simply use the groundstate energy as the free energy.
                if np.isinf(Phi):
                    Phi =  2*np.sum(eigVals *fermiFunc(par,eigVals))


        # <H*>*
            effEnergy = np.sum(eigVals*2.0*fermi)

        # <H>*
        # first the one particle part:
            onePartEnergy = 2.0*np.sum(denMatFull * hamNonInt)
        # second the two particle part:
        # calculate the two particle density matrix for the impurity block:
            twoPartUpUp = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
            twoPartUpDn = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
            for ii in range(par.NimpOrbs):
                for jj in range(par.NimpOrbs):
                    for kk in range(par.NimpOrbs):
                        for ll in range(par.NimpOrbs):
                            twoPartUpUp[ii,jj,kk,ll] = (denMatFull[ii,ll] * denMatFull[jj,kk]
                                                       -denMatFull[ii,kk] * denMatFull[jj,ll])
                            twoPartUpDn[ii,jj,kk,ll] = denMatFull[ii,ll] * denMatFull[jj,kk]
            twoPartEnergy = 0.0
        # factor two of u-matrix cancels because of the spin degeneracy:
        # DnDn and DnUp are not considered...
            for ii in range(par.NimpOrbs):
                for jj in range(par.NimpOrbs):
                    for kk in range(par.NimpOrbs):
                        for ll in range(par.NimpOrbs):
                            twoPartEnergy += par.uMatrix[ii,jj,ll,kk] * (
                                                     twoPartUpUp[ii,jj,kk,ll]
                                                    +twoPartUpDn[ii,jj,kk,ll]
                                                             )

            origEnergy = onePartEnergy + twoPartEnergy
            phiTilde = Phi + origEnergy - effEnergy
        
            print('phiTilde from HF', phiTilde)
            print('occ', np.diag(denMat))
            hFSol['phiTilde'] = phiTilde

            # calculate S^2
            # factor 2 from spin degeneracy                                          
            SzSquare = 2.0*np.trace(denMatFull[:par.NimpOrbs,:par.NimpOrbs])
            for iO in range(par.NimpOrbs):
                for jO in range(par.NimpOrbs):
                    SzSquare += -2.0 * twoPartUpUp[iO,jO,iO,jO] - 2.0 * twoPartUpDn[iO,jO,jO,iO]
            # <S+S- + S-S+>
            SpmmpSquare = 2.0 * np.trace(denMatFull[:par.NimpOrbs,:par.NimpOrbs])
            for iO in range(par.NimpOrbs):
                for jO in range(par.NimpOrbs):
                    SpmmpSquare += -2.0 * twoPartUpDn[iO,jO,iO,jO]
            sSquare = SzSquare + 0.5*SpmmpSquare
            print('sSquare from easyFour', sSquare)
                                     
                    
        
        #print ham
        #hFSol = dict()
        hFSol['eigVals'] = eigVals
        hFSol['eigVecs'] = eigVecs
        hFSol['denMat'] = denMatFull
        #hFSol['hamilLoc'] = np.dot(eigVecs,np.dot(np.diag(eigVals),eigVecs.transpose()))
        hFSol['hamilLoc'] = ham

        #hFSol['phiTilde'] = phiTilde
        

    return hFSol

def solveHartreeFockFixImpOcc(par,parHF,denMat):
    #%%     
    epsdMatrix=  np.diag(par.epsImp)
    
    hamNonInt=onePartMatrix(par.epsBath,par.vBath,epsdMatrix,par.Nbath)
    
    eigValsFirst,eigVecs= np.linalg.eigh(hamNonInt,UPLO='U')

    sig = selfEnergy(par.uMatrix,denMat)

    ham = onePartMatrix(par.epsBath,par.vBath,epsdMatrix+sig,par.Nbath)
    eigVals,eigVecs= np.linalg.eigh(ham,UPLO='U')
    denMat = densityMatrixCorr(eigVals,eigVecs,par.mu,par.beta,par.NimpOrbs)
    

    hFSol['eigVals'] = eigVals
    hFSol['eigVecs'] = eigVecs
    hFSol['denMat'] = denMat
    return hFSol
    
    
def firstGuessHFSubSpace(par):
    #%%     
    #nVals = eigVecsHF.shape[0]
    
    #matrix = eigVecsHF
    #vVec = np.zeros(shape=(nVals))
    
    #vVec[:par.NimpOrbs] = 0.0
    
    coeffsD = np.zeros(shape=(par.NimpOrbs,par.Nbath+1))
    coeffsC1 = np.zeros(shape=(par.NimpOrbs,par.Nbath+1))
    coeffsD[:,0] = 1.0
    coeffsD[:,1:] = 0.02
    #coeffsD /= np.sqrt(np.sum(coeffsD**2)) 
    
    coeffsC1[:,0] = 0.02
    coeffsC1[:,1:] = 1.0

    #coeffsC1 /= np.sqrt(np.sum(coeffsC1**2)) 
    
    
    
    # make basis for uncorrelated space,
    # one for each orbital
    
    coeffsOrb = np.zeros(shape=(par.NimpOrbs,1+par.Nbath,1+par.Nbath))
    for iO in range(par.NimpOrbs):
        W = np.eye(coeffsD.shape[1])
        W[:,0] = coeffsD[iO,:]
        W[:,1] = coeffsC1[iO,:]
        
        q,r = np.linalg.qr(W,mode='complete')
        if q[0,0]<0:
                q = -q
       
        coeffsOrb[iO,:,:] = q 
    for iO in range(par.NimpOrbs):
        coeffsD[iO,:] = coeffsOrb[iO,:,0]
        coeffsC1[iO,:] = coeffsOrb[iO,:,1]

    return coeffsD, coeffsC1

def corrSubHamiltonian(cfg,par,hFSol,coeffsD,coeffsC1):
    #%%     
    #for iO in range(par.NimpOrbs):
    #    coeffsD[iO,:] /= np.sqrt(np.sum(coeffsD[iO,:]**2))    
    #    coeffsC1[iO,:] /= np.sqrt(np.sum(coeffsC1[iO,:]**2))
    
    onePartHamilCorrSub = np.zeros(shape=(2*par.NimpOrbs,2*par.NimpOrbs))
    onePartHamilCorrSubOrb = np.zeros(shape=(par.NimpOrbs,2,2))
    twoPartHamilCorrSub = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
    twoPartHamilAllCorrSub = np.zeros(shape=(2*par.NimpOrbs,2*par.NimpOrbs,2*par.NimpOrbs,2*par.NimpOrbs))
           
    dummy = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs))
    #print 'eigenVec',hFSol['eigVecs'][0,:]
    #print 'ndCoeff',coeffsD[0,:]
    #print 'norm',np.sum(coeffsD[0,:]**2)
    
    for iO in range(par.NimpOrbs):
        dummy[iO,iO] = coeffsD[iO,0]
        #for jO in range(par.NimpOrbs):
        #    dummy[iO,jO] =  np.sum(coeffsD[iO,:]*hFSol['eigVecs'][jO,:])
    
    for ii in range(par.NimpOrbs):
        #dumI = [np.newaxis]
        for jj in range(par.NimpOrbs):
            #dumJ = coeffsD[jj,0][np.newaxis]
            for kk in range(par.NimpOrbs):
                #dumK = coeffsD[kk,0][np.newaxis]
                for ll in range(par.NimpOrbs):
                    #dumL = coeffsD[ll,0][np.newaxis]
                    twoPartHamilCorrSub[ii,jj,kk,ll] = par.uMatrix[ii,jj,kk,ll] * coeffsD[ii,0] * coeffsD[jj,0] * coeffsD[kk,0] * coeffsD[ll,0]
                    #twoPartHamilCorrSub[ii,jj,kk,ll] = np.tensordot(np.tensordot(np.tensordot(np.tensordot(par.uMatrix,dumL,axes=(3,0)),dumK,axes=(2,0)),dumJ,axes=(1,0)),dumI,axes=(0,0))
                    #twoPartCorrLocUpUp[ii,jj,kk,ll] = np.tensordot(np.tensordot(np.tensordot(np.tensordot(twoPart['upup'],dumL,axes=(3,0)),dumK,axes=(2,0)),dumJ,axes=(1,0)),dumI,axes=(0,0))
    coeffsAll = np.zeros(shape=(2*par.NimpOrbs,coeffsD.shape[1]))                
    coeffsAll[:par.NimpOrbs,:]  = coeffsD
    coeffsAll[par.NimpOrbs:,:]  = coeffsC1
    for mm in range(2*par.NimpOrbs):
        for nn in range(2*par.NimpOrbs):
            for oo in range(2*par.NimpOrbs):
                for pp in range(2*par.NimpOrbs):
                    #for ii in range(par.NimpOrbs):
                        #dumI = [np.newaxis]
                        #for jj in range(par.NimpOrbs):
                            #dumJ = coeffsD[jj,0][np.newaxis]
                            #for kk in range(par.NimpOrbs):
                                #dumK = coeffsD[kk,0][np.newaxis]
                                #for ll in range(par.NimpOrbs):
                                    twoPartHamilAllCorrSub[mm,nn,oo,pp] = coeffsAll[mm,0] * coeffsAll[nn,0] * coeffsAll[oo,0] * coeffsAll[pp,0] * par.uMatrix[mm%par.NimpOrbs,nn%par.NimpOrbs,oo%par.NimpOrbs,pp%par.NimpOrbs]
                    
    #print test
    #print dummy
    #for iO in range(par.NimpOrbs):
    #    for jO in range(par.NimpOrbs):
    #        for kO in range(par.NimpOrbs):
    #            for lO in range(par.NimpOrbs):
    #                for iOp in range(par.NimpOrbs):
    #                    for jOp in range(par.NimpOrbs):
    #                        for kOp in range(par.NimpOrbs):
    #                            for lOp in range(par.NimpOrbs):
    #                                
    #                                twoPartHamilCorrSub[iO,jO,kO,lO] += dummy[iO,iOp] * dummy[jO,jOp]  * dummy[kO,kOp]  * dummy[lO,lOp] * par.uMatrix[iOp,jOp,kOp,lOp]
    #print twoPartHamilCorrSub   
    #a=b             
    # double counting: transform HF-hamiltonian into loc. basis,
    # subtract double counting energy sigma und transform back to
    # HF - eigenbasis, then we can transform that to \tilde a
    if cfg['algo']['noHF'] == False:

        hamilLocDC = copy.copy(hFSol['hamilLoc'])    
        #hamilLocDC[:par.NimpOrbs,:par.NimpOrbs] -= selfEnergy(twoPartHamilCorrSub,hFSol['denMat'])
        hamilLocDC[:par.NimpOrbs,:par.NimpOrbs] -= selfEnergy(par.uMatrix,hFSol['denMat'])
        hamilLocDCOrb = np.zeros(shape=(par.NimpOrbs,1+par.Nbath,1+par.Nbath))
        for iO in range(par.NimpOrbs):
            hamilLocDCOrb[iO,0,0] = hamilLocDC[iO,iO]
            hamilLocDCOrb[iO,0,1:] = hamilLocDC[iO,par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath]
            hamilLocDCOrb[iO,1:,0] = hamilLocDC[par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath,iO]
            hamilLocDCOrb[iO,1:,1:] = hamilLocDC[par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath,par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath]
    else:
        hamilLocDCOrb = onePartMatrixOrb(par.epsBath,par.vBath,np.diag(par.epsImp),par.epsBath.shape[1])
        hamilLocDC = onePartMatrix(par.epsBath,par.vBath,np.diag(par.epsImp),par.epsBath.shape[1])
    
    # make a orbital resolved form of hamilLocDC:
     
    for iO in range(par.NimpOrbs):
        onePartHamilCorrSub[iO,iO] = np.dot(coeffsD[iO,:].transpose(),np.dot(hamilLocDCOrb[iO,:,:],coeffsD[iO,:]))
        onePartHamilCorrSub[iO,iO+par.NimpOrbs] = np.dot(coeffsD[iO,:].transpose(),np.dot(hamilLocDCOrb[iO,:,:],coeffsC1[iO,:]))
        onePartHamilCorrSub[iO+par.NimpOrbs,iO] = np.dot(coeffsC1[iO,:].transpose(),np.dot(hamilLocDCOrb[iO,:,:],coeffsD[iO,:]))
        onePartHamilCorrSub[iO+par.NimpOrbs,iO+par.NimpOrbs] = np.dot(coeffsC1[iO,:].transpose(),np.dot(hamilLocDCOrb[iO,:,:],coeffsC1[iO,:]))
        
        onePartHamilCorrSubOrb[iO,0,0] = np.dot(coeffsD[iO,:].transpose(),np.dot(hamilLocDCOrb[iO,:,:],coeffsD[iO,:]))
        onePartHamilCorrSubOrb[iO,0,1] = np.dot(coeffsD[iO,:].transpose(),np.dot(hamilLocDCOrb[iO,:,:],coeffsC1[iO,:]))
        onePartHamilCorrSubOrb[iO,1,0] = np.dot(coeffsC1[iO,:].transpose(),np.dot(hamilLocDCOrb[iO,:,:],coeffsD[iO,:]))
        onePartHamilCorrSubOrb[iO,1,1] = np.dot(coeffsC1[iO,:].transpose(),np.dot(hamilLocDCOrb[iO,:,:],coeffsC1[iO,:]))

    return onePartHamilCorrSub, onePartHamilCorrSubOrb, twoPartHamilCorrSub, twoPartHamilAllCorrSub
   
def unCorrSubHamiltonian(cfg,par,hamilLoc,coeffsD,coeffsC1):
    #%% 

    # make basis for uncorrelated space,
    # one for each orbital
    coeffsOrb = np.zeros(shape=(par.NimpOrbs,1+par.Nbath,1+par.Nbath))
    for iO in range(par.NimpOrbs):
        W = np.eye(coeffsD.shape[1])
        W[:,0] = coeffsD[iO,:]
        W[:,1] = coeffsC1[iO,:]
        q,r = np.linalg.qr(W,mode='complete')
        if q[0,0]<0:
                q = -q
       
        coeffsOrb[iO,:,:] = copy.copy(q)
        #coeffsOrb[iO,:,0] = coeffsD[iO,:]
        #coeffsOrb[iO,:,1] = coeffsC1[iO,:]
    
    # these are orbital diagonal transformations,
    # now we want to build a general transformation
    
    coeffs = np.zeros(shape=(par.NimpOrbs*(1+par.Nbath),par.NimpOrbs*(1+par.Nbath)))
    for iO in range(par.NimpOrbs):
        
        indexD = iO
        indexCFrom = par.NimpOrbs+iO*(par.Nbath)
        indexCTo = par.NimpOrbs+(iO+1)*(par.Nbath)
        # \tilde d - coefficients
        # first all firsts - belonging to d parts
        coeffs[indexD,iO] = coeffsOrb[iO,0,0]
        # then all other parts - belonging to c parts
        coeffs[indexCFrom:indexCTo,iO] = coeffsOrb[iO,1:,0]
        # \tilde c - coefficients
        # first all firsts - belonging to d parts
        coeffs[indexD,iO+par.NimpOrbs] = coeffsOrb[iO,0,1]
        # then all other parts - belonging to c parts
        coeffs[indexCFrom:indexCTo,iO+par.NimpOrbs] = coeffsOrb[iO,1:,1]
        # \tilde rest - coefficients
        # first all firsts - belonging to d parts
        coeffs[indexD,2*par.NimpOrbs+(par.Nbath-1)*iO:2*par.NimpOrbs+(par.Nbath-1)*(iO+1)] = coeffsOrb[iO,0,2:]
        # then all other parts - belonging to c parts
        coeffs[indexCFrom:indexCTo,2*par.NimpOrbs+(par.Nbath-1)*iO:2*par.NimpOrbs+(par.Nbath-1)*(iO+1)] = coeffsOrb[iO,1:,2:]
 
        
    
    # this defines a new basis of the uncorrelated space
    # we want the hamiltonian in this space to be diagonal:

    
    onePartHamilUnCorrSubOrb = np.zeros(shape=(par.NimpOrbs,par.Nbath-1,par.Nbath-1))

    if cfg['algo']['noHF'] == False:
        #onePartHamilUnCorrSub = np.dot(coeffs[:,2*par.NimpOrbs:].transpose(),np.dot(hamilLoc,coeffs[:,2*par.NimpOrbs:]))
        #onePartHamilUnCorrSub = np.zeros(shape=((par.Nbath-1)*par.NimpOrbs,(par.Nbath-1)*par.NimpOrbs))
        hamilLocOrb = np.zeros(shape=(par.NimpOrbs,1+par.Nbath,1+par.Nbath))
        for iO in range(par.NimpOrbs):
            hamilLocOrb[iO,0,0] = hamilLoc[iO,iO]
            hamilLocOrb[iO,0,1:] = hamilLoc[iO,par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath]
            hamilLocOrb[iO,1:,0] = hamilLoc[par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath,iO]
            hamilLocOrb[iO,1:,1:] = hamilLoc[par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath,par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath]
            onePartHamilUnCorrSubOrb[iO,:,:] = np.dot(coeffsOrb[iO,:,2:].transpose(),np.dot(hamilLocOrb[iO,:,:],coeffsOrb[iO,:,2:]))
            

    else:
        hamilLoc = onePartMatrix(par.epsBath,par.vBath,np.diag(par.epsImp),par.epsBath.shape[1])
        hamilLocOrb = onePartMatrixOrb(par.epsBath,par.vBath,np.diag(par.epsImp),par.epsBath.shape[1])
        #onePartHamilUnCorrSub = np.dot(coeffs[:,2*par.NimpOrbs:].transpose(),np.dot(hamilLoc,coeffs[:,2*par.NimpOrbs:]))
        for iO in range(par.NimpOrbs):
            onePartHamilUnCorrSubOrb[iO,:,:] = np.dot(coeffsOrb[iO,:,2:].transpose(),np.dot(hamilLocOrb[iO,:,:],coeffsOrb[iO,:,2:]))
    
    
    return onePartHamilUnCorrSubOrb, coeffsOrb, coeffs

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T
        


   
def calcUpdatedObservables(par,parHF,out,transOldToTilde,corrEigVecs,onePartDensMatrix,densityMatrixOuterWindow,twoPartUpUp,twoPartUpDn,unCorrEigVecs,normD):
    #%%
    
    transformNewLocToOldLoc = np.dot(transOldToTilde,corrEigVecs.transpose())
    print(normD)
    locDensMat = np.dot(np.dot(transformNewLocToOldLoc.transpose(),onePartDensMatrix),transformNewLocToOldLoc)
    locDensMat[:par.NimpOrbs,:par.NimpOrbs] = locDensMat[:par.NimpOrbs,:par.NimpOrbs]
    print('den outer', np.diag(densityMatrixOuterWindow[:par.NimpOrbs,:par.NimpOrbs]))
    print('den inner ED', np.diag(onePartDensMatrix[:par.NimpOrbs,:par.NimpOrbs]))
    print('den inner trans', np.diag(locDensMat[:par.NimpOrbs,:par.NimpOrbs]))
    print('den inner trans + outer ',  np.diag(densityMatrixOuterWindow[:par.NimpOrbs,:par.NimpOrbs] + (locDensMat[:par.NimpOrbs,:par.NimpOrbs])), np.sum(np.diag(densityMatrixOuterWindow[:par.NimpOrbs,:par.NimpOrbs] + (locDensMat[:par.NimpOrbs,:par.NimpOrbs]))))
    
    out['HF'] = dict()
    out['HF']['phiTilde'] = 0.0
    out['HF']['twoPart'] = 0.0
    

    totalLocDenMat = locDensMat*2 + densityMatrixOuterWindow*2

    #dum = transOldToTilde[:par.NimpOrbs,:]
    dum = transformNewLocToOldLoc[:par.NimpOrbs,:par.NimpOrbs]

    twoPartLocUpDn = np.tensordot(np.tensordot(np.tensordot(np.tensordot(twoPartUpDn,dum,axes=(3,0)),dum,axes=(2,0)),dum,axes=(1,0)),dum,axes=(0,0))
    twoPartLocUpUp = np.tensordot(np.tensordot(np.tensordot(np.tensordot(twoPartUpUp,dum,axes=(3,0)),dum,axes=(2,0)),dum,axes=(1,0)),dum,axes=(0,0))
    
    
    out['HF']['twoPartUpUpED'] = twoPartLocUpUp
    out['HF']['twoPartUpDnED'] = twoPartLocUpDn
    out['HF']['onePartED'] = locDensMat
    
    #newFour = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
    easyFourUpDn = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
    easyFourUpUp = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
    #fermi = 1.0 / (np.exp(par.beta*(UnCorrEigVals-par.mu))+1)
    #occMat = np.diag(fermi) 
    for ii in range(par.NimpOrbs):
        for jj in range(par.NimpOrbs):
            for kk in range(par.NimpOrbs):
                for ll in range(par.NimpOrbs):
                    easyFourUpDn[ii,jj,kk,ll] = (densityMatrixOuterWindow[ii,ll]*densityMatrixOuterWindow[jj,kk]
                                            +locDensMat[ii,ll]*densityMatrixOuterWindow[jj,kk]
                                            +locDensMat[jj,kk]*densityMatrixOuterWindow[ii,ll]
                                            +twoPartLocUpDn[ii,jj,kk,ll])
                    easyFourUpUp[ii,jj,kk,ll] = (densityMatrixOuterWindow[ii,ll]*densityMatrixOuterWindow[jj,kk]
                                            -densityMatrixOuterWindow[ii,kk]*densityMatrixOuterWindow[jj,ll]
                                            +locDensMat[ii,ll]*densityMatrixOuterWindow[jj,kk]
                                            +locDensMat[jj,kk]*densityMatrixOuterWindow[ii,ll]
                                            -locDensMat[ii,kk]*densityMatrixOuterWindow[jj,ll]
                                            -locDensMat[jj,ll]*densityMatrixOuterWindow[ii,kk]
                                            +twoPartLocUpUp[ii,jj,kk,ll])
    
    
    sSquare = 0.0
    # calculate <S^2> =< 0.5* (S+S- + S-S+) + Sz^2>                                                   
    # <Sz^2>                                                                                          
    # factor 2 from spin degeneracy
    # totalLocDenMat is allready summed over spin channels                                              
    SzSquare = 0.25*np.trace(totalLocDenMat[:par.NimpOrbs,:par.NimpOrbs])
    for iO in range(par.NimpOrbs):
        for jO in range(par.NimpOrbs):
            SzSquare += -0.5 * easyFourUpUp[iO,jO,iO,jO] - 0.5 * easyFourUpDn[iO,jO,jO,iO]
    # <S+S- + S-S+>                                                                                   
    SpmmpSquare = np.trace(totalLocDenMat[:par.NimpOrbs,:par.NimpOrbs])
    for iO in range(par.NimpOrbs):
        for jO in range(par.NimpOrbs):
            SpmmpSquare += -2.0 * easyFourUpDn[iO,jO,iO,jO]
    sSquare = SzSquare + 0.5*SpmmpSquare

    print('sSquare from easyFour', sSquare)
    
    fac = 0.5*np.sqrt(12.0)
    lplus = np.array([[0, 1.0, 0, -1.0, 0], \
                      [1.0, 0, fac, 0, -1.0], \
                      [0, 1j*fac, 0, 1j*fac, 0 ], \
                      [1.0, 0, -fac, 0, -1.0], \
                      [0, 1.0, 0, -1.0, 0]])
    lminus = np.array([[0, 1.0, 0, 1.0, 0], \
                      [1.0, 0, -1j*fac, 0, 1.0], \
                      [0, fac, 0, -fac, 0 ], \
                      [-1.0, 0, -1j*fac, 0, -1.0], \
                      [0, -1.0, 0, -1.0, 0]])
    lz = np.array([[0, 0, 0, 0, 2.0], \
                      [0, 0, 0, -1.0, 0], \
                      [0, 0, 0, 0, 0], \
                      [0, -1, 0, 0, 0], \
                      [2.0, 0, 0, 0, 0]],dtype=complex)
            
    lSquare = 0.0
    for ii in range(par.NimpOrbs):
        for jj in range(par.NimpOrbs):
            for kk in range(par.NimpOrbs):
                for ll in range(par.NimpOrbs):
                    dummy = (lplus[ii,jj] * lminus[kk,ll] +lminus[ii,jj] * lplus[kk,ll] +lz[ii,jj] * lz[kk,ll])
                    # 0.5 from def. 2.0 from spin deg.
                    lSquare += -0.5*(easyFourUpUp[ii,kk,jj,ll]*2.0) * dummy
                    if ii == kk:
                        # no factor two, because totalLocDenMat is spin-summed
                        lSquare +=  0.5*totalLocDenMat[ii,ll] * dummy
    lSquare = lSquare.real
    print('lSquare from easyFour', lSquare)
    origOnePartHamil = onePartMatrix(par.backup['epsBath'],par.backup['vBath'],np.diag(par.backup['epsImp']),par.backup['epsBath'].shape[1])

    #print 'easy Four', easyFourUpDn
    energyOnePart = np.sum(totalLocDenMat * origOnePartHamil)
    energyTwoPart = 0.0
    for ii in range(par.NimpOrbs):
        for jj in range(par.NimpOrbs):
            for kk in range(par.NimpOrbs):
                for ll in range(par.NimpOrbs):
                    energyTwoPart += par.backup['uMatrix'][ii,jj,ll,kk] *(
                                     easyFourUpDn[ii,jj,kk,ll]
                                   +easyFourUpUp[ii,jj,kk,ll]
                                      )
    energy = energyOnePart + energyTwoPart
    #energy = np.trace(np.dot(totalLocDenMat,origOnePartHamil)) + np.sum(easyFourUpDn * par.backup['uMatrix'])
    
    #print 'full energy', energy
    #print 'easyFour', easyFourUpDn[0,0,0,0]
    out['HF']['phiTilde'] = energy
    out['HF']['twoPartUpUp'] = easyFourUpUp
    out['HF']['twoPartUpDn'] = easyFourUpDn
    
    print('HF+ED PhiTilde',energy)

    out['HF']['<S^2>'] = sSquare
    out['HF']['<L^2>'] = lSquare
    out['HF']['densMatInner'] = densityMatrixOuterWindow*2
    #out['HF']['densMatOuter'] = densityMatrixInnerWindow*2
    out['HF']['densMatInnerED'] = locDensMat*2
    out['HF']['totalDenMat'] = totalLocDenMat
    
    
    return out
