

import numpy as np

import time
import scipy.sparse as sp
import scipy.linalg as spLinAlg
import pickle
import math
import os
from scipy import optimize
import copy
import edvaraux.newIO as io
import sys
import glob
import edvaraux.uMatrix as uMat
import edvaraux.newHartreeFock as harFock
import pyqs.core.fockalgebra.falgebras as pyAlgebra
from subprocess import call
import inspect
import matplotlib.pyplot as plt
import scipy.integrate as spInt
from scipy.interpolate import interp1d
#import parametersMulOrb


def makeParamsClass(cfg):
    class parametersAIM():
        def __init__(self):
            self.NimpOrbs = cfg['aim']['NimpOrbs']
            self.epsImp = np.zeros(shape=(self.NimpOrbs))
            self.epsImp[:] = np.array(cfg['aim']['epsImp'])
            self.numDeg = cfg['aim']['numDeg']
            self.degeneracy = []
    
            for iD in range(self.numDeg):
                self.degeneracy.append([int(ii) for ii in cfg['aim']['deg'+str(iD+1)]])
            self.beta = cfg['aim']['beta']
            self.Bz = cfg['aim']['Bz']
            self.onlyGroundState = cfg['aim']['onlyGroundState']
            self.mu = cfg['aim']['mu']
            self.UImp = cfg['aim']['UImp']
            self.JImp = cfg['aim']['JImp']
            self.gFact = cfg['aim']['gFact']
            self.lsCoupling = cfg['aim']['lsCoupling']
            self.lsMat = self.gFact * uMat.lsCouplingL2Cubic()
            
            
    par = parametersAIM()
    return par
    
def makeParamsClassSlick(cfg):
    class parametersAIM():
        def __init__(self):

            self.beta = cfg['beta']
            self.onlyGroundState = cfg['onlyGroundState']
            self.mu = cfg['mu']
    par = parametersAIM()
    return par

def mats(beta,nMats):
    return np.arange(1,nMats*2,2)*math.pi/beta

def nonIntGFInv(hyb,epsd,iw):

    return (iw - epsd - hyb)

def calcHybCaf(Vk,iw,epsk):
    if epsk.size == 1:
        epsk = epsk[np.newaxis]
        Vk = Vk[np.newaxis]
    denom = ( iw[:,np.newaxis] - epsk[np.newaxis,:] )
    hyb =   np.sum(Vk[np.newaxis,:]**2 / denom, axis=1)
    return hyb    

def costFunction(x0,giwInvOrig,iw,epsd,fac):
    hybEff = calcHybCaf(x0[1],iw,x0[0])
    chi2 =1.0/(iw.size+1.0) * np.sum((iw.imag**(-fac))*(giwInvOrig - nonIntGFInv(hybEff,epsd,iw))*(giwInvOrig - nonIntGFInv(hybEff,epsd,iw)).conjugate())
    chi2 = chi2.real
    #print(chi2)
    return chi2

def firstGuessCafKrauth(par,cfg):
    
    bfgsOpts = {}
    iw = mats(cfg['algo']['betaFit'],cfg['algo']['numIWFit'])

    startV = cfg['algo']['startV']
    startE = cfg['algo']['startE']
    print('\nsearching starting point for bath parameters by Caf Krauth')
    print('random starting points for Caf Krauth fit between {0:g}eV and {1:g}eV for the bath energy and {2:g}eV and {3:g}eV for the hybridization'.format(startE[0],startE[1],startV[0],startV[1]))

    pointEps = np.zeros(par.numDeg)
    pointV = np.zeros(par.numDeg)
    for iD in range(par.numDeg):
        print('deg. orbital',iD)
        minfm = 1e100
        hyb = calcHybCaf(par.vBath[par.degeneracy[iD][0],:],1j*iw,par.epsBath[par.degeneracy[iD][0],:])
        giw = nonIntGFInv(hyb,par.epsImp[par.degeneracy[iD][0]]-par.mu,1j*iw)
        #vVec = np.linspace(startV[0],startV[1],80)
        #for iV in range(vVec.size):
        #    pointS = np.array([0.0,vVec[iV]])
        #    #print(pointS)
        #    cost = costFunction(pointS,giw[:],1j*iw,par.epsImp[par.degeneracy[iD][0]],cfg['algo']['weightFunc'])
        #    #print(cost)
        for ii in range(cfg['algo']['numItersRandFit']):
            pointS = np.array([(np.random.rand()+startE[0]/float(startE[1]-startE[0]))*(startE[1]-startE[0]), (np.random.rand()+startV[0]/float(startV[1]-startV[0]))*(startV[1]-startV[0])])
            #print pointS
            o=optimize.minimize(costFunction,pointS,args=(giw[:],1j*iw,par.epsImp[par.degeneracy[iD][0]],cfg['algo']['weightFunc'],),method='BFGS',options=bfgsOpts)
            
            #out=costFunction(pointS,giw[:],1j*iw,par.epsImp[par.degeneracy[iD][0]],cfg['algo']['weightFunc'])
            #diff = minfm-out

            diff = minfm-o.fun 
            if diff > 0:
                minfm = o.fun
                #minfm = out
                #minPoint = pointS
                
                minPoint = o.x
                print('iteration',ii,'diff', diff)
                if diff < cfg['algo']['maxDeltaFitBreak']:
                    break
            
            
            #fm[ii] = o.fun
            #pm[ii,:] = o.x
        pointV[iD] = minPoint[1]
        pointEps[iD] = minPoint[0]
    return pointV, pointEps


def fermiFunc(par,energies):
    #%% 
    if par.onlyGroundState:
        fermiFunction =  (-np.sign(energies - par.mu)+1)*0.5
    else:
        # this leads to overflows for small betas
        #fermiFunction = 1.0 / ( np.exp((energies-par.mu)*par.beta) + 1 ) 
        # this does not lead to overflow:
        fermiFunction = 0.5* (1.0 - np.tanh((energies-par.mu)*par.beta*0.5))
    return fermiFunction
    
def makeNameStrings(par):
    #%% 
    # make list of names:
    spins = ['up','dn']
    orbs = [str(ii+1) for ii in range(par.NimpOrbs)]
    baths = [str(ii+1) for ii in range(par.Nbath)]
    nameStrings = []
    for iS in spins:
        for iO in orbs:
            nameStrings.extend(['+imp'+iO+'_'+iS,'-imp'+iO+'_'+iS])
        for iO in orbs:
            for iB in baths:
                nameStrings.extend(['+'+iO+'_'+iB+iS,'-'+iO+'_'+iB+iS])
    return nameStrings
    
def makeNameStringsNew(par):
    #%% 
    # make list of names:
    spins = ['up','dn']
    orbs = [str(ii+1) for ii in range(par.NimpOrbs)]
    baths = [str(ii+1) for ii in range(par.Nbath)]
    nameStrings = []
    for iS in spins:
        for iO in orbs:
            nameStrings.extend(['imp'+iO+'_'+iS])
        for iO in orbs:
            for iB in baths:
                nameStrings.extend([''+iO+'_'+iB+iS])
    return nameStrings

def get_diagblocks_structure(blk_pattern):
    #%% 
    from collections import Counter

    # return a dict with 'items' of blk_pattern as its keys and their corresponding occurency in blk_pattern as its values
    hits = Counter(blk_pattern.tolist())
    # create a list of hits's values sorted by hits's keys, i.e, hitlist represents the dimensions of diagonal blocks in the right order
    hitlist = [value for (key, value) in sorted(hits.items())]
    keylist = [key for (key, value) in sorted(hits.items())]
    
    return keylist, hitlist 
    
def idiagblocks(sparse_matrix, blk_ind, ret_blkpos = False):
    #%%     

    
    for iBlock in range(len(blk_ind)-1):

        if ret_blkpos:
            yield sparse_matrix[blk_ind[iBlock]:blk_ind[iBlock+1], blk_ind[iBlock]:blk_ind[iBlock+1]], {"rows": rows, "cols": cols}
        else:
            yield sparse_matrix[blk_ind[iBlock]:blk_ind[iBlock+1], blk_ind[iBlock]:blk_ind[iBlock+1]]
        
#def idiagblocksSet(sparse_matrix, low, high,ret_blkpos = False):
#    #%%     
#    rows = [0, 0]
#    cols = [0, 0]

#    for blk_shape in blk_pattern:
#        rows[1] += blk_shape
#        cols[1] += blk_shape

#        if ret_blkpos:
#            yield sparse_matrix[rows[0]:rows[1], cols[0]:cols[1]], {"rows": rows, "cols": cols}
#        else:
#            yield sparse_matrix[rows[0]:rows[1], cols[0]:cols[1]]
        
#        rows[0] += blk_shape
#        cols[0] += blk_shape
    
def set_ham_pure_twoPart(par,ham):
    print('setting two particle part of Hamiltonian...',end=' ',flush=True)
    start = time.time()
    # initialize Hamiltonian:
    spStr = ['_up', '_dn']
    H = 0.0*copy.copy(ham['fullLocal'])
    version = 'slow' 
    for ii in range(par.NimpOrbs):
        for jj in range(par.NimpOrbs):
            for kk in range(par.NimpOrbs):
                for ll in range(par.NimpOrbs):
                    for iS in range(2):
                        for jS in range(2):
                            # coulomb matrix
                            if par.uMatrix[ii,jj,ll,kk] != 0:
                                
                                if version == 'slow': # seems actually to be faster ?!?
                                    H =  ( H
                                        + 0.5 * par.uMatrix[ii,jj,ll,kk]
                                        *(ham['oper']['+imp'+str(ii+1)+spStr[iS]]
                                        * ham['oper']['+imp'+str(jj+1)+spStr[jS]]
                                        * ham['oper']['-imp'+str(kk+1)+spStr[jS]]
                                        * ham['oper']['-imp'+str(ll+1)+spStr[iS]])
                                            )
                                elif version == 'fast':
                                    if iS == jS:
                                        if ii > jj:
                                            cPcP = -ham['oper']['++'+str(jj)+'_'+str(ii)+spStr[iS]+spStr[jS]]
                                        else:
                                            cPcP =  ham['oper']['++'+str(ii)+'_'+str(jj)+spStr[iS]+spStr[jS]]
                                        if ll > kk:
                                            cMcM = -ham['oper']['++'+str(kk)+'_'+str(ll)+spStr[iS]+spStr[jS]].transpose()
                                        else:
                                            cMcM =  ham['oper']['++'+str(ll)+'_'+str(kk)+spStr[iS]+spStr[jS]].transpose()
                                    elif iS > jS:
                                        cPcP = -ham['oper']['++'+str(jj)+'_'+str(ii)+spStr[jS]+spStr[iS]]
                                        cMcM = -ham['oper']['++'+str(kk)+'_'+str(ll)+spStr[jS]+spStr[iS]].transpose()
                                    
                                    else:
                                        cPcP = ham['oper']['++'+str(ii)+'_'+str(jj)+spStr[iS]+spStr[jS]]
                                        cMcM = ham['oper']['++'+str(ll)+'_'+str(kk)+spStr[iS]+spStr[jS]].transpose()
                                    H =  ( H
                                        + 0.5 * par.uMatrix[ii,jj,ll,kk]
                                        * (cPcP * cMcM)
                                            )
    print('took {:1.2f} sec'.format(time.time() - start),flush=True)
    return H
                                #H =  ( H
                                #    + 0.5 * par.uMatrix[ii,jj,ll,kk]
                                #    * ham['oper']['++'+str(ii)+'_'+str(jj)+spStr[iS]+spStr[jS]]
                                #    * ham['oper']['++'+str(ll)+'_'+str(kk)+spStr[iS]+spStr[jS]].transpose()
                                #        )
    
def set_ham_pure(par,ham):
    #%%
    spStr = ['_up', '_dn']
    #  part:

    H = copy.copy(ham['fullLocal'])
    for ii in range(par.NimpOrbs):
        for jj in range(par.Nbath):
            # bath diagonal part
            H = H + par.epsBath[ii,jj] *( ham['bathUp'][ii][jj] + ham['bathDn'][ii][jj] )  
            
            
            H = H + ( par.vBath[ii,jj] * ham['hybUp'][ii][jj]
                     + par.vBath[ii,jj].conjugate() * ham['hybUp'][ii][jj].transpose()
                     )
            # spin down
            H = H + ( par.vBath[ii,jj] * ham['hybDn'][ii][jj]
                    + par.vBath[ii,jj].conjugate() * ham['hybDn'][ii][jj].transpose()
                     )
    for ii in range(par.NimpOrbs):
        # one-particle on site energy
        H =  ( H
            +(par.epsImp[ii] - par.mu) 
            * ( ham['oper']['+imp'+str(ii+1)+'_up']*ham['oper']['-imp'+str(ii+1)+'_up']
            +   ham['oper']['+imp'+str(ii+1)+'_dn']*ham['oper']['-imp'+str(ii+1)+'_dn'] )
            )
        # magnetic field in z-direction on atom
        # factor 0.5 from m_s = 0.5
        H =  ( H
            +(0.5*par.Bz)
            * ( ham['oper']['+imp'+str(ii+1)+'_up']*ham['oper']['-imp'+str(ii+1)+'_up']
            -   ham['oper']['+imp'+str(ii+1)+'_dn']*ham['oper']['-imp'+str(ii+1)+'_dn'] )
            )
        if par.NimpOrbs == 5:
            lzMat,lpMat,lmMat = uMat.matrixWrapper(par)
             # factor 0.5 from g-factor of L is half of g factor of S
            for jj in range(par.NimpOrbs):
                H =  ( H +
                    0.5*par.Bz* lzMat[jj,ii].real
                    *( ham['oper']['+imp'+str(ii+1)+'_up'] * ham['oper']['-imp'+str(jj+1)+'_up']
                    +ham['oper']['+imp'+str(ii+1)+'_dn'] * ham['oper']['-imp'+str(jj+1)+'_dn'])
                    )
        
 
    if par.lsCoupling:
       
        if (par.NimpOrbs != 5):
            raise Exception('ls coupling only implemented for 5 orbitals')
        for ii in range(par.NimpOrbs):
            for jj in range(par.NimpOrbs):
                for iS in range(len(spStr)):
                    for jS in range(len(spStr)):
                        H = H + par.lsMat[iS*par.NimpOrbs+ii,jS*par.NimpOrbs+jj]*(ham['oper']['+imp'+str(ii+1)+spStr[iS]]*ham['oper']['-imp'+str(jj+1)+spStr[jS]]).tocsr()

    #print(H.toarray()[np.abs(H.toarray().imag])!=0)
    # if par.optOnly == True, then ham['fullLocal'] allready contains the 
    # two-particle part of the hamiltonian
       
    if hasattr(par,'twoPart'):
        H = H + par.twoPart
    else:
        H = H + set_ham_pure_twoPart(par,ham)
        


    return H



def set_ham_SIAM(cfg, par, mycar,ham):
    #%% 
    start = time.time()
    # for this step we set the uMatrix to zero. in this way, the setting 
    # of the dummy hamiltonian to get the blockstructure is faster
    # this only works if the specified quantum numbers are not corrupted by
    # the u-matrix
    #print('killing uMatrix for finding block structue. Lets hope youre quantum numbers are compatible with the u-matrix!!')
    #uMat = copy.copy(par.uMatrix)
    #par.uMatrix[...] = 0.0    
    #H = set_ham_pure(par,ham)
    #par.uMatrix = uMat   
    end = time.time()
    start = time.time()
    if cfg['algo']['show']: print("finding blocks",flush=True)
    #Hsort, diagblk_qnum, diagblk_dim, qnum = find_blocks(H,par,ham)
    diagblk_qnum, diagblk_dim, qnum, qnumNames = find_blocks_WOHam(cfg,par,ham)
    print("found "+str(len(diagblk_qnum))+ ' blocks, maximum dimension is '+str(np.max(diagblk_dim)),flush=True)
    end = time.time()
    if cfg['algo']['show']: 
        print( "time finding blocks", end-start,flush=True)
    print('copying mycar... ',end='',flush=True)
    blockmycar = copy.deepcopy(mycar)
    print('finished',flush=True)
    print('restructurung mycar... ',end='',flush=True)
    blockmycar.restructure_matrix(qnum)
    print('finished',flush=True)
    return diagblk_qnum, diagblk_dim, qnum,qnumNames, blockmycar
    
    
def set_ham_SIAMnotBlock(par, mycar,ham):
    #%% 
    H = set_ham_pure(par,ham)
    
    return H
    
    
def preCalculateOperators(car,par):
    #%% 
    ham = dict()
    #start = time.time()
    spStr = ['_up', '_dn']

    if par.lsCoupling:
        ham['fullLocal'] = sp.csr_matrix((2**(par.NfStates), 2**(par.NfStates)), dtype=complex)
    else:
        ham['fullLocal'] = sp.csr_matrix((2**(par.NfStates), 2**(par.NfStates)), dtype=float)
    ham['oper'] = dict()
    #car["+imp"+str(0+1)+"_up"].matrix[:,:]
    #carer[(0,'creation')].matrix[:,:]
    for ii in range(par.NimpOrbs):

        
        ham['oper']['+imp'+str(ii+1)+'_up'] = car["+imp"+str(ii+1)+"_up"].matrix[:,:]
        ham['oper']['+imp'+str(ii+1)+'_dn'] = car["+imp"+str(ii+1)+"_dn"].matrix[:,:]
        ham['oper']['-imp'+str(ii+1)+'_up'] = car["-imp"+str(ii+1)+"_up"].matrix[:,:]
        ham['oper']['-imp'+str(ii+1)+'_dn'] = car["-imp"+str(ii+1)+"_dn"].matrix[:,:]
        
        ham['oper']['+'+str(ii)+'_up'] = ham['oper']['+imp'+str(ii+1)+'_up']
        ham['oper']['+'+str(ii)+'_dn'] = ham['oper']['+imp'+str(ii+1)+'_dn']
        ham['oper']['-'+str(ii)+'_up'] = ham['oper']['-imp'+str(ii+1)+'_up']
        ham['oper']['-'+str(ii)+'_dn'] = ham['oper']['-imp'+str(ii+1)+'_dn']

    for ii in range(par.NimpOrbs):
        
        for jj in range(par.Nbath):
            ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_up'] = car['+'+str(ii+1)+'_'+str(jj+1)+'up'].matrix[:,:]
            ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_dn'] = car['+'+str(ii+1)+'_'+str(jj+1)+'dn'].matrix[:,:]
            ham['oper']['-bath'+str(ii+1)+'_'+str(jj+1)+'_up'] = car['-'+str(ii+1)+'_'+str(jj+1)+'up'].matrix[:,:]
            ham['oper']['-bath'+str(ii+1)+'_'+str(jj+1)+'_dn'] = car['-'+str(ii+1)+'_'+str(jj+1)+'dn'].matrix[:,:]
            
            # here same values but different keys...
            # convenient for calculating two-particle density matrix
            # this does not use more memory!
            ham['oper']['+'+str(par.NimpOrbs+jj+ii*par.Nbath)+'_up'] = ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_up']
            ham['oper']['+'+str(par.NimpOrbs+jj+ii*par.Nbath)+'_dn'] = ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_dn']
            ham['oper']['-'+str(par.NimpOrbs+jj+ii*par.Nbath)+'_up'] = ham['oper']['-bath'+str(ii+1)+'_'+str(jj+1)+'_up']
            ham['oper']['-'+str(par.NimpOrbs+jj+ii*par.Nbath)+'_dn'] = ham['oper']['-bath'+str(ii+1)+'_'+str(jj+1)+'_dn']

    # precalculate c+isig c+jsig' and c-isigc-jsig'
    if par.Nbath > 0:
        preCalcNum = 2*par.NimpOrbs
    else:
        preCalcNum = par.NimpOrbs
    for ii in range(preCalcNum):
        
        for jj in range(preCalcNum):
            #print 'ii jj', ii, jj
            for iS in spStr:
                for jS in spStr:
                    
                    if (iS == jS) and (ii > jj):
                        #print same spins
                        pass
                    elif (iS == '_dn') and (jS == '_up') :
                        #print spin one dn, spin t
                        pass
                    else:
                        pass
                        ham['oper']['++'+str(ii)+'_'+str(jj)+iS+jS] = ham['oper']['+'+str(ii)+iS] * ham['oper']['+'+str(jj)+jS]
                        #car[(0,'creation')].matrix[:,:]*car[(0,'creation')].matrix[:,:]
                        #car["+imp"+str(ii+1)+iS].matrix[:,:] * car["-imp"+str(ii+1)+jS].matrix[:,:]

    ham['bathUp'] = []
    ham['bathDn'] = []
    ham['hybUp'] = []
    ham['hybDn'] = []
    
    for ii in range(par.NimpOrbs):
        
        dummyBathUp = []
        dummyBathDn = []
        dummyhybUp = []
        dummyhybDn = []
        for jj in range(par.Nbath):
            dummyBathUp.append(ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_up'] * ham['oper']['-bath'+str(ii+1)+'_'+str(jj+1)+'_up'])
            dummyBathDn.append(ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_dn'] * ham['oper']['-bath'+str(ii+1)+'_'+str(jj+1)+'_dn'])
            dummyhybUp.append( ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up'])
            dummyhybDn.append( ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_dn'] * ham['oper']['-imp'+str(ii+1)+'_dn'])
            #ham['bathUp'+str(ii)+'_'+str(jj)] = ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_up'] * ham['oper']['-bath'+str(ii+1)+'_'+str(jj+1)+'_up']
            #ham['bathDn'+str(ii)+'_'+str(jj)] = ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_dn'] * ham['oper']['-bath'+str(ii+1)+'_'+str(jj+1)+'_dn']
            #ham['hybUp'+str(ii)+'_'+str(jj)] = ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up']
            #ham['hybDn'+str(ii)+'_'+str(jj)] = ham['oper']['+bath'+str(ii+1)+'_'+str(jj+1)+'_dn'] * ham['oper']['-imp'+str(ii+1)+'_dn']
        ham['bathUp'].append(dummyBathUp)
        ham['bathDn'].append(dummyBathDn)
        ham['hybUp'].append(dummyhybUp)
        ham['hybDn'].append(dummyhybDn)
    
    ham['N_ops'] = []
    for ii in range(par.NfStates):
        ham['N_ops'].append((car[(ii, "creation")].matrix[:,:] * car[(ii, "annihilation")].matrix[:,:]).diagonal())
    ham['S_orbsUp'] = []
    ham['S_orbsDn'] = []
    if par.Nbath > 0:
        for ii in range(par.NimpOrbs):
            ham['S_orbsUp'].append(sum(ham['bathUp'][ii][:]).diagonal() + ham['oper']['+imp'+str(ii+1)+'_up']*ham['oper']['-imp'+str(ii+1)+'_up'].diagonal())
            ham['S_orbsDn'].append(sum(ham['bathDn'][ii][:]).diagonal() + ham['oper']['+imp'+str(ii+1)+'_dn']*ham['oper']['-imp'+str(ii+1)+'_dn'].diagonal())

    #print 'calc N_ops OperDIags',time.time() - start
    
    return ham
    
    
def preCalculateTwoPart(ham,par):
    #%% 
    print('precalculating two part... ',end=' ',flush=True)
    start = time.time()
    spStr = ['_up', '_dn']
    for ii in range(par.NimpOrbs):
        for jj in range(par.NimpOrbs):
            for kk in range(par.NimpOrbs):
                for ll in range(par.NimpOrbs):
                    for iS in range(2):
                        for jS in range(2):
                            # coulomb matrix
                            if par.uMatrix[ii,jj,ll,kk] != 0:
                                #ham['fullLocal'] =  ( ham['fullLocal']
                                #    + 0.5 * par.uMatrix[ii,jj,ll,kk]
                                #    * ham['oper']['+imp'+str(ii+1)+spStr[iS]]
                                #    * ham['oper']['+imp'+str(jj+1)+spStr[jS]]
                                #    * ham['oper']['-imp'+str(kk+1)+spStr[jS]]
                                #    * ham['oper']['-imp'+str(ll+1)+spStr[iS]]
                                #              )
                                if iS == jS:
                                    if ii > jj:
                                        cPcP = -ham['oper']['++'+str(jj)+'_'+str(ii)+spStr[iS]+spStr[jS]]
                                    else:
                                        cPcP =  ham['oper']['++'+str(ii)+'_'+str(jj)+spStr[iS]+spStr[jS]]
                                    if ll > kk:
                                        cMcM = -ham['oper']['++'+str(kk)+'_'+str(ll)+spStr[iS]+spStr[jS]].transpose()
                                    else:
                                        cMcM =  ham['oper']['++'+str(ll)+'_'+str(kk)+spStr[iS]+spStr[jS]].transpose()
                                elif iS > jS:
                                    cPcP = -ham['oper']['++'+str(jj)+'_'+str(ii)+spStr[jS]+spStr[iS]]
                                    cMcM = -ham['oper']['++'+str(kk)+'_'+str(ll)+spStr[jS]+spStr[iS]].transpose()
                                    
                                else:
                                    cPcP = ham['oper']['++'+str(ii)+'_'+str(jj)+spStr[iS]+spStr[jS]]
                                    cMcM = ham['oper']['++'+str(ll)+'_'+str(kk)+spStr[iS]+spStr[jS]].transpose()
                                ham['fullLocal'] =  ( ham['fullLocal']
                                    + 0.5 * par.uMatrix[ii,jj,ll,kk]
                                    * (cPcP * cMcM)
                                        )  
    print('took {:1.2f} sec'.format(time.time() - start),flush=True)
    return ham


def find_blocks_WOHam(cfg,par,ham):
    #%% 
    N_ops = ham['N_ops']    
    NfStates = par.NfStates
    # names of conserved quantum numbers
    
    names = ["N"]
    if not par.lsCoupling:
        names.extend(["Ndown", "Nup"])
    else:
        pass
        #names.extend(["Nbathdown", "Nbathup"])
    if cfg['aim']['terms'] == 'dens':
        for ii in range(par.NimpOrbs):
            names.extend(["S_orbs"+str(ii+1)+"Up","S_orbs"+str(ii+1)+"Dn"])

    qnum = np.zeros(2 ** NfStates, dtype = {"names": names, "formats": ["int"] * len(names)})
    # fill in the quantum number values corresponding to actual matrix structure

    qnum["N"] = sum(N_ops)
    if not par.lsCoupling:
        qnum["Ndown"] = sum(N_ops[NfStates//2:])
        qnum["Nup"] = sum(N_ops[:NfStates//2])
    else:
        pass
        #qnum["Nbathdown"] = sum(N_ops[NfStates//2:][par.NimpOrbs:])
        #qnum["Nbathup"] = sum(N_ops[:NfStates//2][par.NimpOrbs:])
    if cfg['aim']['terms'] == 'dens':
        for ii in range(par.NimpOrbs):
            qnum["S_orbs"+str(ii+1)+"Up"] = ham['S_orbsUp'][0]
            qnum["S_orbs"+str(ii+1)+"Dn"] = ham['S_orbsDn'][0]

    diagblk_qnum, diagblk_dim = get_diagblocks_structure(qnum)
       
    return diagblk_qnum, diagblk_dim, qnum, names

    
def find_blocks(H,par,ham):
    #%% 
    N_ops = ham['N_ops']    
    NfStates = par.NfStates
    # names of conserved quantum numbers
    
    names = ["N", "Ndown", "Nup"]
    if par.terms == 'dens':
        for ii in range(par.NimpOrbs):
            names.extend(["S_orbs"+str(ii+1)+"Up","S_orbs"+str(ii+1)+"Dn"])
    #print names
    #conserved_qnums = len(names)
    # names of state occupation numbers (for an inblock structuring)
    #names = []
    #names.extend(["".join(["N", str(ii)]) for ii in range(NfStates - 1, -1, -1)])
    #print names
    #conserved_qnums = len(names)
    # numpy.ndarray of named quantum number tuples
    qnum = np.zeros(2 ** NfStates, dtype = {"names": names, "formats": ["int"] * len(names)})
    # fill in the quantum number values corresponding to actual matrix structure
    #print 'n_ops', N_ops
    #print 'sum(n_ops)',sum(N_ops)
    qnum["N"] = sum(N_ops)
    qnum["Ndown"] = sum(N_ops[NfStates//2:])
    qnum["Nup"] = sum(N_ops[:NfStates//2])
    if par.terms == 'dens':
        for ii in range(par.NimpOrbs):
            qnum["S_orbs"+str(ii+1)+"Up"] = ham['S_orbsUp'][0]
            qnum["S_orbs"+str(ii+1)+"Dn"] = ham['S_orbsDn'][0]

    #for ii, name in enumerate(names[conserved_qnums:]):
    #    qnum[name] = N_ops[NfStates - 1 - ii]

    perm = np.argsort(qnum)
    permmat = sp.coo_matrix(([1] * len(perm), (range(len(perm)), perm)))
    #print (permmat*H*permmat.transpose()).toarray()
    #diagblk_qnum, diagblk_dim = get_diagblocks_structure(qnum[["N", "Nup", "Ndown"]])
    diagblk_qnum, diagblk_dim = get_diagblocks_structure(qnum)
    #diagblk_qnum, diagblk_dim = get_diagblocks_structure(qnum[["N"]]) 
    Hsort= permmat * H * permmat.transpose()
    
    return Hsort, diagblk_qnum, diagblk_dim, qnum


def find_blocks_with_QN(qnums,diagblk_qnum,diagblk_dim):
    Dim = len(diagblk_dim)
    indizes = np.zeros(Dim+1,dtype=int)
    indizes[1:] = np.cumsum(diagblk_dim)
    ind = []
    for iB in range(len(diagblk_qnum)):
        boolVec = np.zeros(shape=(len(qnums)),dtype=bool)
        
        for iQ in range(len(qnums)):
            boolVec[iQ] = (diagblk_qnum[iB][iQ] == qnums[iQ]) or (qnums[iQ] == None)

        if np.all(boolVec):
            ind.append(iB)
    low = []
    high = []
    for ii in range(len(ind)):
        low.append(indizes[ind[ii]])
        high.append(indizes[ind[ii]+1])
    return low,high

def findGroundState(eVa,qnums):

    minList = [np.min(blockEner) for blockEner in eVa]
    # lowest energy
    E0Prime = min(minList)
    # now find all blocks in which groundstate energy exists
    minBlockIndex = [jj for jj, vv in enumerate(minList) if np.abs(vv - E0Prime)<1e-12]
    E0 = []
    E0Index = []
    NG = qnums[minBlockIndex[0]][0]
    # now get all groundstate energies (and their indices) in the blocks
    for ii in minBlockIndex:
        dummy = []
        for jj in range(eVa[ii].size):
            if np.abs(eVa[ii][jj]-E0Prime)<1e-12:
                E0.append(eVa[ii][jj])
                dummy.extend([jj])
                #E0Index.append([ii, jj])
        E0Index.append(dummy)
    # get all states which are thermodynamically relevant:
    return E0, minBlockIndex,E0Index, NG
    
def diagonalize_set_blocks(H,low,high,method,params):
    
    if method != 'full' and  method != 'sparse':
        raise NameError('error, choose method correctly')
    
    count = 0
    totalTime = 0.
    eigValsList = []
    eigVecsList = []
    # clean up the directory: check if eigenvectors with current dateString are in dir
    # if not, delete all eigenvectors in temp dir
    #if len(glob.glob('/work/mschueler/temp/eigenVecs/eV*'+par.dateString+'.pkl')) == 0:
    #     for pklFiles in glob.glob('/work/mschueler/temp/eigenVecs/eV*.pkl'):
    #         os.remove(pklFiles) 

    for iB in range(len(low)):
        block = H[low[iB]:high[iB],low[iB]:high[iB]]
        
        if block.shape[0] < params['Nthreshhold'] or method == 'full':
            #eigVals,eigVecs = np.linalg.eigh(block.toarray())
            eigVals,eigVecs = spLinAlg.eigh(block.toarray())
        else:
            # try to get previous eigenvectors of groundstate as starting guess:
            #try:
            #    
            #    f = open(par.eigVecName+'/eV'+str(count)+par.dateString+'.pkl','rb')
            #     v0 = pickle.load(f)
            #    f.close()
            #    firstGuess = True
            #except:
            #    
            #    firstGuess = False
            firstGuess = False
            for ii in range(params['valUpdateIter']):
                numEigVals = params['Nvalues'] + ii * params['valUpdateValues']
                try:
                    #print numEigVals
                    #if firstGuess == True:
                    #    eigVals,eigVecs = sp.linalg.eigsh(block,k=numEigVals,which='SA',maxiter=params.arnoldiIters,tol=params.tol,ncv=2*numEigVals+1,v0=v0)
                    #else:
                    eigVals,eigVecs = sp.linalg.eigsh(block,k=numEigVals,which='SA',maxiter=params['arnoldiIters'],tol=params['tol'],ncv=2*numEigVals+1)
                    # if everything is fine, we can stop here, else the loop is
                    # continued with more eigenvalues
                    break
                except sp.linalg.eigen.ArpackNoConvergence:
                    pass
                    # do not print!
                    #print( 'Lanczos did not converge, now ',numEigVals+(ii+1) * params.valUpdateValues,'eigenvalues')
            # save eigenvector of one eigenvalue:
            #f = open(par.eigVecName+'/eV'+str(count)+par.dateString+'.pkl','wb')
            
            #pickle.dump(eigVecs[:,0],f)
            #f.close()
        eigValsList.append(eigVals)
        eigVecsList.append(eigVecs)

        count += 1
    return eigValsList, eigVecsList


    
def diagonalize_blocks(H,diagblk_qnum,diagblk_dim,diagblk_ind,method,params):
    #%%     
    if method != 'full' and  method != 'sparse':
        raise NameError('error, choose method correctly')
    
    count = 0
    totalTime = 0.
    eigValsList = []
    eigVecsList = []
    # clean up the directory: check if eigenvectors with current dateString are in dir
    # if not, delete all eigenvectors in temp dir
    #if len(glob.glob('/work/mschueler/temp/eigenVecs/eV*'+par.dateString+'.pkl')) == 0:
    #     for pklFiles in glob.glob('/work/mschueler/temp/eigenVecs/eV*.pkl'):
    #         os.remove(pklFiles) 

    for block in idiagblocks(H, diagblk_ind):
        
        if diagblk_dim[count] < params['Nthreshhold'] or method == 'full':
            #print('diag full block with part. num', diagblk_qnum[count][0],'dim.',block.shape[0],flush=True,end=' ')
            #eigVals,eigVecs = np.linalg.eigh(block.toarray())
            eigVals,eigVecs = spLinAlg.eigh(block.toarray())
            #print(block)
            #print(eigVals)
            #print(diagblk_dim[count])
            #print(eigVecs[0,:])
        #elif (diagblk_qnum[count][1] != 5) and (diagblk_qnum[count][1] != 5):
        #    print('skipping')
        #    eigVals = np.zeros(shape=(1))
        #    eigVecs = np.zeros(shape=(1))
            
        else:
            start = time.time()
            print('diag sparse block with part. num', diagblk_qnum[count][0],'dim.',block.shape[0],flush=True,end=' ')
            firstGuess = False
            for ii in range(params['valUpdateIter']):
                
                numEigVals = params['Nvalues'] + ii * params['valUpdateValues']
                print(numEigVals,flush=True,end=' ')
                try:
                    #print numEigVals
                    #if firstGuess == True:
                    #    eigVals,eigVecs = sp.linalg.eigsh(block,k=numEigVals,which='SA',maxiter=params.arnoldiIters,tol=params.tol,ncv=2*numEigVals+1,v0=v0)
                    #else:
                    eigVals,eigVecs = sp.linalg.eigsh(block,k=numEigVals,which='SA',maxiter=params['arnoldiIters'],tol=params['tol'],ncv=2*numEigVals+1)
                    # if everything is fine, we can stop here, else the loop is
                    # continued with more eigenvalues
                    break
                except sp.linalg.eigen.ArpackNoConvergence as partialRes:
                    pass
                    
                    eigVals = partialRes.eigenvalues
                    eigVecs = partialRes.eigenvectors
                    if eigVals.size == 0:
                        pass
                        #raise Exception('no converged eigenvalues found, increase [ed][arnoldiIters] or number of [ed][Nvalues]!')
                    else:
                        break
                    # do not print!
                    #print(sp.linalg.eigen.ArpackNoConvergence.eigenvalues)
                    #print( 'Lanczos did not converge, now',numEigVals+(ii+1) * params['valUpdateValues'],'eigenvalues',end=' ')
                # save eigenvector of one eigenvalue:
                #f = open(par.eigVecName+'/eV'+str(count)+par.dateString+'.pkl','wb')
            print('took {:1.2f}s'.format(time.time()-start))
            #pickle.dump(eigVecs[:,0],f)
            #f.close()
        eigValsList.append(eigVals)
        eigVecsList.append(eigVecs)

        count += 1
    return eigValsList, eigVecsList

def diagonalize_blocks_part(H,diagblk_qnum,diagblk_dim,method,params,nPart):
    #%%     
    if method != 'full' and  method != 'sparse':
        raise NameError('error, choose method correctly')
    
    count = 0
    totalTime = 0.
    eigValsList = []
    eigVecsList = []
    # clean up the directory: check if eigenvectors with current dateString are in dir
    # if not, delete all eigenvectors in temp dir
    #if len(glob.glob('/work/mschueler/temp/eigenVecs/eV*'+par.dateString+'.pkl')) == 0:
    #     for pklFiles in glob.glob('/work/mschueler/temp/eigenVecs/eV*.pkl'):
    #         os.remove(pklFiles) 

    for block in idiagblocks(H, diagblk_dim):
        print('diagonalizing block with part. number', diagblk_qnum[count][0],'dim.',block.shape[0],flush=True)
        if np.any(diagblk_qnum[count][0] == nPart) == False:
            # do not diagonalize particle number blocks
            # set eigenvalue to biggest number, so we don't mistaken it for the groundstate
            eigVals = np.finfo('d').max
            eigVecs = None
        else:
            if diagblk_dim[count] < params['Nthreshhold'] or method == 'full':
                #eigVals,eigVecs = np.linalg.eigh(block.toarray())
                eigVals,eigVecs = spLinAlg.eigh(block.toarray())
            else:
                # try to get previous eigenvectors of groundstate as starting guess:
                #try:
                #    
                #    f = open(par.eigVecName+'/eV'+str(count)+par.dateString+'.pkl','rb')
                #     v0 = pickle.load(f)
                #    f.close()
                #    firstGuess = True
                #except:
                #    
                #    firstGuess = False
                firstGuess = False
                for ii in range(params['valUpdateIter']):
                    numEigVals = params['Nvalues'] + ii * params['valUpdateValues']
                    try:
                        #print numEigVals
                        #if firstGuess == True:
                        #    eigVals,eigVecs = sp.linalg.eigsh(block,k=numEigVals,which='SA',maxiter=params.arnoldiIters,tol=params.tol,ncv=2*numEigVals+1,v0=v0)
                        #else:
                        eigVals,eigVecs = sp.linalg.eigsh(block,k=numEigVals,which='SA',maxiter=params['arnoldiIters'],tol=params['tol'],ncv=2*numEigVals+1)
                        # if everything is fine, we can stop here, else the loop is
                        # continued with more eigenvalues
                        break
                    except sp.linalg.eigen.ArpackNoConvergence:
                        pass
                        # do not print!
                        #print( 'Lanczos did not converge, now ',numEigVals+(ii+1) * params.valUpdateValues,'eigenvalues')
                # save eigenvector of one eigenvalue:
                #f = open(par.eigVecName+'/eV'+str(count)+par.dateString+'.pkl','wb')
                
                #pickle.dump(eigVecs[:,0],f)
                #f.close()
        eigValsList.append(eigVals)
        eigVecsList.append(eigVecs)

        count += 1
    return eigValsList, eigVecsList


def partitionFunctionSpectra(par,ham,eigVals):

    E0, minBlockIndex,E0Index, NG =findGroundState(eigVals,ham['diagblk_qnum'])

    ZZ1BlockTil = []
    ZZ1Til = 0
    thermoImportance = []

    # loop over subspaces, do not sum, because we need these factors also to
    # evaluate the expectaion values. we can reuse them there
    for iBlock in range(len(eigVals)):
        ZZ1BlockTil.append(np.exp(-par.beta * ( eigVals[iBlock] - E0[0] - par.mu*ham['diagblk_qnum'][iBlock][0] )))
        thermoImportance.append(np.sum(ZZ1BlockTil[iBlock]))
        sumBlock = np.sum(ZZ1BlockTil[iBlock])
        ZZ1Til += sumBlock
        #if abs(ham['diagblk_qnum'][iBlock][0] - NG) < (par.PMBlock+1):

    return ZZ1Til,ZZ1BlockTil,E0,minBlockIndex,E0Index,thermoImportance

def partitionFunction(par,ham,eigVals,eigVecs,cfg):
    if par.onlyGroundState:
        E0, minBlockIndex, E0Index, NG =findGroundState(eigVals,ham['diagblk_qnum'])
        eigVecsLocal = copy.copy(eigVecs)
        # save the corresponding eigenvalues and vectors in lists
        eigVecsLocal = [eigVecsLocal[iB] for iB in minBlockIndex]
        # set all eigenvectors to zero, which belong to eigenvalues which 
        # are not groundstate
        for iB in range(len(minBlockIndex)):
            boolList = np.ones(eigVecsLocal[iB].shape[1],dtype=bool)
            for kk in range(eigVecsLocal[iB].shape[1]):
                if np.all(kk != np.array(E0Index[iB][:])):
                    boolList[kk] = False
                    #eigVecs[iB][:,kk] = 0.0
            eigVecsLocal[iB] = eigVecsLocal[iB][:,boolList]
                    
        Phi = E0[0]
        
        
        ZZ1Block = minBlockIndex # dirty misuse of variable
        ZZ1BlockTil = minBlockIndex # dirty misuse of variable
        ZZ1 = []
        ZZ1Til = len(E0)
        ZZ1TilRes = []
        thermoImportance = []
        nQnum = len(ham['diagblk_qnum'][0])
        print('qnum blocks with groundstate importance:',len(minBlockIndex))
        for ii in range(len(minBlockIndex)):
            for iQ in range(nQnum):
                print( ham['qnumNames'][iQ],ham['diagblk_qnum'][minBlockIndex[ii]][iQ],end=' ')
            print('')
    else:
        E0, minBlockIndex,E0Index, NG =findGroundState(eigVals,ham['diagblk_qnum'])
        eigVecsLocal = copy.copy(eigVecs)
        ZZ1BlockTil = []
        
        ZZ1BlockTilClean = []
        thermoImportance = []
        ZZ1Til = 0
        ZZ1TilRes = 0
        # loop over subspaces, do not sum, because we need these factors also to
        # evaluate the expectaion values. we can reuse them there
        diffFactor = E0[0] - par.mu * par.mu*ham['diagblk_qnum'][minBlockIndex[0]][0]
        diffFactor = E0[0]
        #diffFactor = 0.0
        for iBlock in range(len(eigVals)):
            boolList = np.exp(-par.beta * ( eigVals[iBlock] - diffFactor - par.mu*ham['diagblk_qnum'][iBlock][0] )) > cfg['ed']['betaCutOffExpecValue']
            #print( boolList)
            eigVals[iBlock] = eigVals[iBlock][boolList]
            eigVecsLocal[iBlock] = eigVecsLocal[iBlock][:,boolList]

        
        
        for iBlock in range(len(eigVals)):
            ZZ1BlockTil.append(np.exp(-par.beta * ( eigVals[iBlock] - diffFactor - par.mu*ham['diagblk_qnum'][iBlock][0] )))
            thermoImportance.append(np.sum(ZZ1BlockTil[iBlock]))
            sumBlock = np.sum(ZZ1BlockTil[iBlock])
            ZZ1Til += sumBlock
            #if abs(ham['diagblk_qnum'][iBlock][0] - NG) < (par.PMBlock+1):
            ZZ1TilRes += sumBlock
            thermoImportance[iBlock]
        print('qnum blocks with thermodynamic importance:')
    
        nQnum = len(ham['diagblk_qnum'][0])
        for iBlock in range(len(eigVals)):
            
            if thermoImportance[iBlock] > 0.0:
                for iQ in range(nQnum):
                    print( ham['qnumNames'][iQ],ham['diagblk_qnum'][iBlock][iQ],end=' ')
                print('with {:1.1e}'.format(thermoImportance[iBlock]))
            
    
    
        
        Phi = (-1/par.beta*np.log(ZZ1Til)) + diffFactor
    return ZZ1Til,ZZ1BlockTil,Phi,NG,E0,eigVecsLocal,thermoImportance,minBlockIndex,E0Index


    
def expecBlockCore(dic):

    matBlock = dic['matBlock']
    eigVecs = dic['eigVecs']
    tempFactor = dic['tempFactor']
    #print( "block %d calculating" % eigVecs.shape)
    return np.sum(tempFactor*np.sum(matBlock.dot(eigVecs.conjugate())*eigVecs,axis=0))


def calcExpecValue(eigVecs,diagblk_ind,ZZ1Block,ZZ1,mat,par):
    #%%  
 
    if par.onlyGroundState:

        expect = 0.0
        count = 0        
        
        for iBlock in ZZ1Block:     
            matBlock = mat[diagblk_ind[iBlock]:diagblk_ind[iBlock+1],diagblk_ind[iBlock]:diagblk_ind[iBlock+1]]
            expect += np.sum(np.sum(matBlock.dot(eigVecs[count].conjugate())*eigVecs[count],axis=0))
            count += 1

        expect = expect / float(ZZ1)
            
        
    
    else:
        #aa = 0
        aa = 0.0
        Dim = len(eigVecs)
        
        #for i in range(10):
        #    t = Thread(target=sleeper, args=(i,))
        #    t.start()

        dicList = []
        for iBlock in range(Dim):
            matBlock = mat[diagblk_ind[iBlock]:diagblk_ind[iBlock+1],diagblk_ind[iBlock]:diagblk_ind[iBlock+1]]
            tempFactor = ZZ1Block[iBlock]
            dic = dict()
            dic['matBlock'] = matBlock
            dic['tempFactor'] = tempFactor
            dic['eigVecs']= eigVecs[iBlock]
            dicList.append(dic)

        for ii in dicList:
            aa += expecBlockCore(ii)
        expect = aa/ZZ1

    return expect

  
def calcExpecValueRes(eigVecs,diagblk_ind,diagblk_qnum,NG,ZZ1Block,ZZ1,mat,cfg,par,thermo):
    #%% 
    #print('going into Res')
    #mat[0:2,0:2]
    # if restric is False, fall back to normal calcExpecValue
    if cfg['ed']['restric'] == False:
        expect = calcExpecValue(eigVecs,diagblk_ind,ZZ1Block,ZZ1,mat,par)
    else:

        if par.onlyGroundState:
            Dim = len(eigVecs)    
            expect = 0.0
            count = 0       
            
            for iBlock in ZZ1Block:
                
                matBlock = mat[diagblk_ind[iBlock]:diagblk_ind[iBlock+1],diagblk_ind[iBlock]:diagblk_ind[iBlock+1]]
                expect += np.sum(np.sum(matBlock.dot(eigVecs[count].conjugate())*eigVecs[count],axis=0))
                count += 1
            expect = expect / float(ZZ1)
    
        else:
            aa = 0.0
            Dim = len(eigVecs)
            #print NG
            for iBlock in range(Dim):
                if thermo[iBlock] > cfg['ed']['betaCutOffExpecValue']:
                    #mat[0:2,0:2]
                    matBlock = mat[diagblk_ind[iBlock]:diagblk_ind[iBlock+1],diagblk_ind[iBlock]:diagblk_ind[iBlock+1]]
                    # tempFactor is exp(-beta E_n)
                    tempFactor = ZZ1Block[iBlock]
                    
                    # do the calculation only for those eigenvalues/eigenvectors
                    # for which exp(-beta * E_n ) is bigger than a certain threshhold
                    #boolean = tempFactor>par.betaCutOffExpecValue
                    #eViB = eigVecs[iBlock][:,boolean]
                    #tempFac = tempFactor[boolean]
                    aa += np.sum(tempFactor*np.sum(matBlock.dot(eigVecs[iBlock].conjugate())*eigVecs[iBlock],axis=0))
            
    
            expect = aa/ZZ1

    return expect
    
    

    
    
def findAndSetBlockCar(cfg, par,nameStrings):
    #%%
    
    fail = True
    if fail == True:
        print("creating eff algebra...",end=" ",flush=True)

        mycar = pyAlgebra.CAR_Algebra(par.NfStates, nameStrings)

        #mycar = pyAlgebra.CAR_Algebra(par.NfStates)
        #print(par.NfStates)
        #mycar[(0,'creation')].matrix[:,:]*mycar[(0,'creation')].matrix[:,:]
        #mycar["+imp"+str(0+1)+"_up"].matrix[:,:]
        #mycar["+imp1_up"]
        #mycar["-imp1_up"]
        
        #newcar["+imp"+str(0+1)+"_up"].matrix[:,:]
        print("finished",flush=True)
        print("precalculating eff Operators... ",end='',flush=True)
        ham = preCalculateOperators(mycar,par)
        print("finished",flush=True)
        print("finding block-structure by setting up eff hamiltonian...",end=" ",flush=True)
        #vBar = np.sqrt(np.sum(vTilde**2,axis=1))[:]
        #par.epsBath =  -0.05 * np.ones(shape=(par.NimpOrbs,par.Nbath))
        #par.vBath = 0.2*np.ones(shape=par.epsBath.shape)
        start = time.time()
        
        

        diagblk_qnum, diagblk_dim, qnum,qnumNames, blockCar = set_ham_SIAM(cfg,par,mycar,ham)
        
        print('took {:1.2f} sec'.format(time.time() - start),flush=True)
        
        print("reprecalculating eff Operators in Block structure...",end=" ",flush=True)
        ham = preCalculateOperators(blockCar,par)
        print("finished",flush=True)
        ham['qnumNames'] = qnumNames
        ham['diagblk_qnum'] = diagblk_qnum
        ham['diagblk_dim'] = diagblk_dim
        ham['diagblk_ind'] = np.zeros(shape=(len(diagblk_dim)+1),dtype=int)
        ham['diagblk_ind'][1:] = np.cumsum(diagblk_dim)
        ham['diagblk_ind'] = list(ham['diagblk_ind'])
        
        # discarding blocks which are not in specified particle number range
        if True:
            nPart = np.arange(cfg['ed']['Nmin'],cfg['ed']['Nmax']+1)   
            
            dummyQnum = copy.copy(ham['diagblk_qnum'])
            dummyDim = copy.copy(ham['diagblk_dim'])
            dummyInd = copy.copy(ham['diagblk_ind'])
            
            
            ham['diagblk_qnum'] = []
            ham['diagblk_dim'] = []
            ham['diagblk_ind'] = []
    
            for iB in range(len(dummyQnum)):
                if np.any(nPart == dummyQnum[iB][0]):
                    ham['diagblk_qnum'].append(dummyQnum[iB])
                    ham['diagblk_dim'].append(dummyDim[iB])
                    if len(ham['diagblk_ind']) == 0:
                        ham['diagblk_ind'].extend(dummyInd[iB:iB+2])
                    else:
                        ham['diagblk_ind'].pop()
                        ham['diagblk_ind'].extend(dummyInd[iB:iB+2])
        
    if False:
        ham = preCalculateTwoPart(ham,par)
    return ham, blockCar


def calcHyb(epsBath, vBath, idelta, numW):

    if len(epsBath.shape) < 2:
        epsBath = epsBath[np.newaxis,:]
        vBath = vBath[np.newaxis,:]
    minEnergy = np.min(epsBath)-0.5-3*idelta-20.0
    maxEnergy = np.max(epsBath)+0.5+3*idelta+20.0
    energy = np.linspace(minEnergy,maxEnergy,num=numW);
    Delta = np.sum(vBath[np.newaxis,:,:]**2 / (epsBath[np.newaxis,:,:] - energy[:,np.newaxis,np.newaxis]-1j*idelta),axis=2)
    return energy, Delta

def calcHybFixEnergy(epsBath, vBath, idelta, energy):

    if len(epsBath.shape) < 2:
        epsBath = epsBath[np.newaxis,:]
        vBath = vBath[np.newaxis,:]

    Delta = np.sum(vBath[np.newaxis,:,:]**2 / (epsBath[np.newaxis,:,:] - energy[:,np.newaxis,np.newaxis]-1j*idelta),axis=2)
    return energy, Delta
    
def fitWrap(idelta,epsFits):
    def fitFunc(energyX,*vFits):
        output = 'list'
        if len(vFits)>1:
            vFits = np.array([vFits])
        else:
            output = 'array'
            vFits = vFits[0]

        energyFit,DeltaFit = calcHybFixEnergy(epsFits, vFits, idelta, energyX)
        
        if output == 'list':
            DeltaFit = list(DeltaFit[:,0].imag)
        

        return DeltaFit
    return fitFunc   
    
def fitHyb(params):
    
    energy, Delta, _ = io.readHyb(params)
    # do a interpolation of the data:
    indices = np.arange(energy.size)
    indicesCutDum = indices[energy<=params['epsMax']]
    indicesCut = indicesCutDum[energy[indicesCutDum]>=params['epsMin']]

    nOrbs = Delta.shape[1]
    vFit = np.zeros(shape=(nOrbs,params['nDisk']))
    epsFit = np.zeros(shape=(nOrbs,params['nDisk']))
    for iO in range(nOrbs):
        epsFit[iO,:] = np.linspace(params['epsMin'],params['epsMax'],num=params['nDisk'],endpoint=True)
        funcLog = fitWrap(params['idelta'],epsFit[iO,:][np.newaxis,:])
        #print epsFit.shape
        optLog,_=optimize.curve_fit(f=funcLog,xdata=energy[indicesCut],ydata=Delta[indicesCut,iO].imag,p0=np.ones(shape=(1,epsFit[iO,:].size)))
        
        vFit[iO,:] = optLog
        # historix version in which the fit was done empirically
#        energy, _, Delta = io.readHyb(parReal)
#        # very sophisticated magic parameters :P
#        mHeight = 0.0313
#        cHeight = 0.0
#    
#        nOrbs = Delta.shape[1]
#        vFit = np.zeros(shape=(nOrbs,parReal.nDisk))
#        epsFit = np.zeros(shape=(nOrbs,parReal.nDisk))
#        for iO in range(nOrbs):
#            epsFit[iO,:] = np.linspace(parReal.epsMin,parReal.epsMax,num=parReal.nDisk,endpoint=True)
#            dFit = epsFit[iO,1] - epsFit[iO,0]
#            vPre = np.interp(epsFit[iO,:], energy, Delta[:,iO].imag )
#            vFit[iO,:] = np.sqrt(vPre) / np.sqrt(mHeight/dFit - cHeight)/10
        # calculate crystal field from real part of hybridization:
        # calculate hybridization function fom fit:
    energyFit, DeltaFit = calcHyb(epsFit, vFit, 0.1, Delta.shape[0])
    # the crystal field is now the mean of the Fit and the ipnut data:
    epsD = -np.mean(DeltaFit.real+Delta.real,axis=0)
    if params['checkByPlot']:
        if True:
            plt.figure(1)
            plt.subplot(2,1,1)
            plt.title('hybridization function from fitted params')
            plt.plot(energyFit,DeltaFit.imag,'r')
            plt.plot(energy,Delta.imag,'k')
            plt.xlabel('w')
            plt.ylabel('hyb')
            plt.subplot(2,1,2)
            plt.title('fitted parameters')
            for iO in range(5):
                plt.plot(epsFit[iO,:],vFit[iO,:])
            plt.plot(epsFit[0,:],np.mean(vFit,axis=0),'k',linewidth=4)
            plt.xlabel('epsBath')
            plt.ylabel('vBath')
            plt.show()
        else:
            plt.figure(1)
            vCaf2 = np.zeros(shape=(nOrbs,1))
            epsCaf2 = np.zeros(shape=(nOrbs,1))
            epsCaf2[:,0] = np.array([-0.80424,-0.5098, -0.2338, -0.5098, -0.80424 ])
            vCaf2[:,0] = np.array([0.77418,0.75519, 0.66205, 0.75519, 0.77418 ])
            
            vCaf1 = np.zeros(shape=(nOrbs,1))
            epsCaf1 = np.zeros(shape=(nOrbs,1))
            epsCaf1[:,0] = np.array([-1.4705699,-1.17976669, -0.37844178, -1.17976669, -1.47056991 ])
            vCaf1[:,0] = np.array([1.07978167,1.12272427, 0.82161704,1.12272427 ,1.07978167 ])
            
            vCaf0 = np.zeros(shape=(nOrbs,1))
            epsCaf0 = np.zeros(shape=(nOrbs,1))
            epsCaf0[:,0] = np.array([-2.23985172,-2.13174041,-1.00891807 ,-2.13174041 ,-2.23985172  ])
            vCaf0[:,0] = np.array([1.53234188,1.48070023,1.23516064 ,1.48070023 ,1.53234188 ])
            
            energyCaf2, DeltaCaf2 = calcHyb(epsCaf2, vCaf2, 0.1, Delta.shape[0])
            energyCaf1, DeltaCaf1 = calcHyb(epsCaf1, vCaf1, 0.1, Delta.shape[0])
            energyCaf0, DeltaCaf0 = calcHyb(epsCaf0, vCaf0, 0.1, Delta.shape[0])
            
            plt.subplot(3,1,1)
            plt.title('hybridization function from fitted params')
            plt.plot(energyFit,DeltaFit.imag,'r')
            plt.plot(energyCaf2,DeltaCaf2.imag,'g')
            plt.plot(energyCaf1,DeltaCaf1.imag,'m')
            plt.plot(energyCaf0,DeltaCaf0.imag,'c')
            plt.plot(energy,Delta.imag,'k')
            plt.legend(loc='best')
            plt.xlabel('w')
            plt.ylabel('hyb')
            plt.subplot(3,1,2)
            plt.title('hybridization function from fitted params')
            for iO in range(epsD.size):
                plt.plot(energyFit,DeltaFit[:,iO].real+epsD[iO],'r')
                plt.plot(energyCaf2,DeltaCaf2[:,iO].real+epsD[iO],'g')
                plt.plot(energyCaf1,DeltaCaf1[:,iO].real+epsD[iO],'m')
                plt.plot(energyCaf0,DeltaCaf0[:,iO].real+epsD[iO],'c')
            plt.plot(energy,-Delta.real,'k')
            plt.xlabel('w')
            plt.ylabel('hyb')
            plt.subplot(3,1,3)
            plt.title('fitted parameters')
            for iO in range(5):
                plt.plot(epsFit[iO,:],vFit[iO,:])
            plt.plot(epsFit[0,:],np.mean(vFit,axis=0),'k',linewidth=4)
            plt.xlabel('epsBath')
            plt.ylabel('vBath')
            plt.show()

    return epsFit, vFit, epsD
   
def setRealPar(cfg,par,parEff):
    #%% 
    if cfg['aim']['readRealMat'] and cfg['aim']['epsBathRange']:
        raise Exception('do not set [aim][epsBathRange] and [aim][readrealMat] simultanously to True')
    if cfg['aim']['readRealMat']:
        #parReal = parametersMulOrb.parametersrealMat()
        print('fitting hybridization function...')
        par.epsBath, par.vBath, crystalField = fitHyb(cfg['realistichyb'])

        par.epsBath = par.epsBath# + par.mu
        #print par.epsBath
        par.NimpOrbs = par.epsBath.shape[0]
        parEff.NimpOrbs = par.epsBath.shape[0]
        par.NimpOrbs = par.epsBath.shape[0]
        
        # put par.epsBath and par.vBath into cfg, so that it is saved in the end
        for iO in range(par.NimpOrbs):
            cfg['aim']['epsBath'+str(iO+1)] = list(par.epsBath[iO,:])
            cfg['aim']['vBath'+str(iO+1)] = list(par.vBath[iO,:])
            
            
        #if par.uOrbs != parReal.onlyBands:
        #    print('overriding par.uOrbs and parEff.uOrbs by parReal.onlyBands!')
        #    par.uOrbs = cfg['realistichyb']['onlyBands']
        #    parEff.uOrbs = cfg['realistichyb']['onlyBands']
        parEff.NfStates = 2*parEff.Nbath*parEff.NimpOrbs + 2*parEff.NimpOrbs
        print('loading hybridization file \''+str(cfg['realistichyb']['filename'])+'\' with '+str(par.NimpOrbs)+' orbitals')
        #if (len(cfg['realistichyb']['onlyBands']) != len(par.uOrbs) ):
        #    raise Exception('number of orbitals in u-matrix (uOrbs) does not fit to number of hybridization bands (onlyBands)')
        print(cfg['aim']['calculateCrystalField'])
        if cfg['aim']['calculateCrystalField']:
            print('adding epsImp and crystal field: epsD =',crystalField)
            par.epsImp += -crystalField
            parEff.epsImp += -crystalField
    elif cfg['aim']['epsBathRange']:
        par.epsBath = np.zeros(shape=(par.NimpOrbs,cfg['aim']['Nbath']))
        for iO in range(par.NimpOrbs):
            par.epsBath[iO,:] = np.linspace(cfg['aim']['epsBathRangeCenter']-cfg['aim']['epsBathRangeBandWidth']/2.0,cfg['aim']['epsBathRangeCenter']+cfg['aim']['epsBathRangeBandWidth']/2.0,cfg['aim']['Nbath'])
        if cfg['aim']['vBath'] == 'simpleV':
            par.vBath = cfg['aim']['vBathRange'] * np.ones(shape=(par.NimpOrbs,cfg['aim']['Nbath']))
        elif cfg['aim']['vBath'] == 'flatDOS':
            par.vBath = cfg['aim']['vBathRange'] * np.ones(shape=(par.NimpOrbs,cfg['aim']['Nbath']))/np.sqrt(cfg['aim']['Nbath'])

        
        
        # check if hybridization function coincides with flat DOS:
        #energyFit, DeltaFit = calcHyb(par.epsBath, par.vBath, 0.05, 2000)
        #plt.figure(1)        
        #plt.plot(energyFit,DeltaFit[:,0].imag,'r')
        #plt.plot(energyFit, cfg['aim']['vBathRange']**2/20.0*np.pi*np.ones(energyFit.size),'k')
        #plt.show()
        #exit()
    par.Nbath = par.epsBath.shape[1]    

    return cfg, par, parEff
def fitFuncWrapNew(pole,cfg):
    #%% 
    if cfg['algo']['funcSystem'] == 'leg':
        valFuncs = np.polynomial.legendre.legval
    elif cfg['algo']['funcSystem'] == 'cheby':
        valFuncs = np.polynomial.chebyshev.chebval
    # todo
    def fitFunc(x, *coeffs):

        coeffs = list(coeffs)
        if len(coeffs) == 1:
            coeffs = coeffs[0]
        vTilde = np.zeros(x.size)
        chebyCoeffs = coeffs

#        if cfg['algo']['fitGunSch']:
#            gunschCoeffsP = chebyCoeffs[2*cfg['algo']['legendreOrder']]
#            gunschCoeffsM = chebyCoeffs[2*cfg['algo']['legendreOrder']+1]
#            vTildeP = 1.0/(gunschCoeffsP+x[x>pole])
#            vTildeM = 1.0/(gunschCoeffsM+x[x<=pole])
#            vTilde += np.hstack((vTildeM,vTildeP))        
        
        if cfg['algo']['legendreOrder'] > 0:
            if cfg['algo']['chebyPole']:
                chebyCoeffsP = chebyCoeffs[:cfg['algo']['legendreOrder']]
                chebyCoeffsM = chebyCoeffs[cfg['algo']['legendreOrder']:]
                dummyP = np.linspace(-1.0,1.0,(x[x>pole]).size)
                dummyM = np.linspace(-1.0,1.0,(x[x<=pole]).size)
                vTildeP = valFuncs(dummyP,chebyCoeffsP)
                vTildeM = valFuncs(dummyM,chebyCoeffsM)
                vTilde += np.hstack((vTildeM,vTildeP))
                
            else:
                dummyX = np.linspace(-1.0,1.0,x.size)
                vTilde += valFuncs(dummyX,chebyCoeffs)
            
        return vTilde
    return fitFunc
    
def vecToPoint(cfg,par,coeffsD,coeffsC1):
    #%%     
    
    if par.numDeg == 0:
        if cfg['algo']['fitLegendre']:
            
            optD = np.zeros(shape=(par.NimpOrbs,par.legendreOrder*2))
            optC1 = np.zeros(shape=(par.NimpOrbs,par.legendreOrder*2))
            for iO in range(par.NimpOrbs):
    
                optD[iO,:],_=optimize.curve_fit(f=fitFuncWrapNew(par.mu,cfg),xdata=par.epsBath[iO,:],ydata=coeffsD[iO,1:],p0=np.ones(shape=2*cfg['algo']['legendreOrder']))
                optC1[iO,:],_=optimize.curve_fit(f=fitFuncWrapNew(par.mu,cfg),xdata=par.epsBath[iO,:],ydata=coeffsC1[iO,1:],p0=np.ones(shape=2*cfg['algo']['legendreOrder']))
            if cfg['algo']['optOnlyBath']:
                point = np.zeros(shape=(par.NimpOrbs*(2*cfg['algo']['legendreOrder'])))
                for iO in range(par.NimpOrbs):
                    point[iO*(2*cfg['algo']['legendreOrder']):(iO+1)*(2*cfg['algo']['legendreOrder'])] = optC1[iO,:]
                
            else:
                point = np.zeros(shape=(par.NimpOrbs*(2+4*cfg['algo']['legendreOrder'])))
                for iO in range(par.NimpOrbs):
                    point[iO*(2+4*cfg['algo']['legendreOrder'])] = coeffsD[iO,0]
                    point[iO*(2+4*cfg['algo']['legendreOrder'])+1:iO*(2+4*cfg['algo']['legendreOrder'])+1+2*cfg['algo']['legendreOrder']] = optD[iO,:]
                    point[iO*(2+4*cfg['algo']['legendreOrder'])+1+2*cfg['algo']['legendreOrder']] = coeffsC1[iO,0]
                    point[iO*(2+4*cfg['algo']['legendreOrder'])+2+2*cfg['algo']['legendreOrder']:iO*(2+4*cfg['algo']['legendreOrder'])+2+4*cfg['algo']['legendreOrder']] = optC1[iO,:]
        #        if cfg['algo']['fitGunSch']:
        #            raise Exception('not implemented')
        #        else:
        point = np.hstack((np.reshape(coeffsD,coeffsD.size),np.reshape(coeffsC1,coeffsC1.size)))
    else:
        if cfg['algo']['fitLegendre']:
            optD = np.zeros(shape=(par.numDeg,cfg['algo']['legendreOrder']*2))
            optC1 = np.zeros(shape=(par.numDeg,cfg['algo']['legendreOrder']*2))
            for iD in range(par.numDeg):
    
                optD[iD,:],_=optimize.curve_fit(f=fitFuncWrapNew(par.mu,cfg),xdata=par.epsBath[par.degeneracy[iD][0],:],ydata=coeffsD[par.degeneracy[iD][0],1:],p0=np.ones(shape=2*cfg['algo']['legendreOrder']))
                optC1[iD,:],_=optimize.curve_fit(f=fitFuncWrapNew(par.mu,cfg),xdata=par.epsBath[par.degeneracy[iD][0],:],ydata=coeffsC1[par.degeneracy[iD][0],1:],p0=np.ones(shape=2*cfg['algo']['legendreOrder']))
            if cfg['algo']['optOnlyBath']:
                point = np.zeros(shape=(par.numDeg*(2*cfg['algo']['legendreOrder'])))
                for iD in range(par.numDeg):
                    point[iD*(2*cfg['algo']['legendreOrder']):(iD+1)*(2*cfg['algo']['legendreOrder'])] = optC1[iD,:]
            else:
                point = np.zeros(shape=(par.numDeg*(2+4*cfg['algo']['legendreOrder'])))
                for iD in range(par.numDeg):
                    point[iD*(2+4*cfg['algo']['legendreOrder'])] = coeffsD[par.degeneracy[iD][0],0]
                    point[iD*(2+4*cfg['algo']['legendreOrder'])+1:iD*(2+4*cfg['algo']['legendreOrder'])+1+2*cfg['algo']['legendreOrder']] = optD[iD,:]
                    point[iD*(2+4*cfg['algo']['legendreOrder'])+1+2*cfg['algo']['legendreOrder']] = coeffsC1[par.degeneracy[iD][0],0]
                    point[iD*(2+4*cfg['algo']['legendreOrder'])+2+2*cfg['algo']['legendreOrder']:iD*(2+4*cfg['algo']['legendreOrder'])+2+4*cfg['algo']['legendreOrder']] = optC1[iD,:]               
        else:
            if cfg['algo']['optOnlyBath']:
                point = np.zeros(shape=(par.numDeg*(par.Nbath)))
                for iD in range(par.numDeg):
                    point[iD*(par.Nbath):(iD+1)*(par.Nbath)] = coeffsC1[par.degeneracy[iD][0],1:]
            else:
                point = np.zeros(shape=(2*par.numDeg*(1+par.Nbath)))
                for iD in range(par.numDeg):
                    point[iD*(par.Nbath+1):(iD+1)*(par.Nbath+1)] = coeffsD[par.degeneracy[iD][0],:]
                    point[par.numDeg*(1+par.Nbath)+iD*(par.Nbath+1):par.numDeg*(1+par.Nbath)+(iD+1)*(par.Nbath+1)] = coeffsC1[par.degeneracy[iD][0],:]

        
    return point 
    
    
def varVectorToParams(point,cfg,par):
    #%% 
    #cfg['algo']['optOnlyBath'] = False
    if par.numDeg == 0:
        if cfg['algo']['optOnlyBath']:
            epsTilde = par.epsImp[:,np.newaxis]
            vTilde = point[0:1*par.NimpOrbs][:,np.newaxis]
            epsBath = point[1*par.NimpOrbs:2*par.NimpOrbs][:,np.newaxis]
        else:
            epsTilde = point[0:par.NimpOrbs][:,np.newaxis]
            # this is actually vBar!!!!!!!!!!!!!!!!!!!
            vTilde = point[par.NimpOrbs:2*par.NimpOrbs][:,np.newaxis]
            epsBath = point[2*par.NimpOrbs:3*par.NimpOrbs][:,np.newaxis]
    else:
        epsTilde = np.zeros(shape=(par.NimpOrbs,1))
        vTilde = np.zeros(shape=(par.NimpOrbs,1))
        epsBath = np.zeros(shape=(par.NimpOrbs,1))
        if cfg['algo']['optOnlyBath']:
            epsTilde = par.epsImp[:,np.newaxis]
            for iD in range(par.numDeg):
                for iOD in range(len(par.degeneracy[iD])):
                    vTilde[par.degeneracy[iD][iOD],0] = point[iD]
            for iD in range(par.numDeg):
                for iOD in range(len(par.degeneracy[iD])):
                    epsBath[par.degeneracy[iD][iOD],0] = point[iD+par.numDeg]   
        else:
            for iD in range(par.numDeg):
                for iOD in range(len(par.degeneracy[iD])):
                    epsTilde[par.degeneracy[iD][iOD],0] = point[iD]
            # this is actually vBar!!!!!!!!!!!!!!!!!!!
            
    
            
            for iD in range(par.numDeg):
                for iOD in range(len(par.degeneracy[iD])):
                    vTilde[par.degeneracy[iD][iOD],0] = point[iD+par.numDeg]
                    
            
            for iD in range(par.numDeg):
                for iOD in range(len(par.degeneracy[iD])):
                    epsBath[par.degeneracy[iD][iOD],0] = point[iD+2*par.numDeg]        
    #cfg['algo']['optOnlyBath'] = True
    return epsTilde, vTilde, epsBath
    
def varVectorToParamswU(point,par):
    #%% 
    if par.numDeg == 0:
        epsTilde = point[0:par.NimpOrbs][:,np.newaxis]
        # this is actually vBar!!!!!!!!!!!!!!!!!!!
        vTilde = point[par.NimpOrbs:2*par.NimpOrbs][:,np.newaxis]
        epsBath = point[2*par.NimpOrbs:3*par.NimpOrbs][:,np.newaxis]
        U = point[3*par.NimpOrbs]
    else:
        epsTilde = np.zeros(shape=(par.NimpOrbs,1))
        for iD in range(par.numDeg):
            for iOD in range(len(par.degeneracy[iD])):
                epsTilde[par.degeneracy[iD][iOD],0] = point[iD]
        # this is actually vBar!!!!!!!!!!!!!!!!!!!
        vTilde = np.zeros(shape=(par.NimpOrbs,1))
        for iD in range(par.numDeg):
            for iOD in range(len(par.degeneracy[iD])):
                vTilde[par.degeneracy[iD][iOD],0] = point[iD+par.numDeg]
                
        epsBath = np.zeros(shape=(par.NimpOrbs,1))
        for iD in range(par.numDeg):
            for iOD in range(len(par.degeneracy[iD])):
                epsBath[par.degeneracy[iD][iOD],0] = point[iD+2*par.numDeg]        
        U = point[3*par.numDeg]
    return epsTilde, vTilde, epsBath, U

def paramsToVarVector(par,epsTilde,vTilde,epsBath):
    #%%     
    if par.numDeg == 0:
        point = np.zeros(shape=(3*par.NimpOrbs))
        point[0:par.NimpOrbs] = epsTilde[:,0]
        point[par.NimpOrbs:2*par.NimpOrbs] = vTilde[:,0]
        point[2*par.NimpOrbs:3*par.NimpOrbs] = epsBath[:,0]
    else:
        point = np.zeros(shape=(3*par.numDeg))
        for iD in range(par.numDeg):
            point[iD] = epsTilde[par.degeneracy[iD][0],0]
            point[iD+par.numDeg] = vTilde[par.degeneracy[iD][0],0]
            point[iD+2*par.numDeg] = epsBath[par.degeneracy[iD][0],0]     
        
    return point
    
def paramsToVarVectorNew(par):
    #%%     
    if par.numDeg == 0:
        point = np.zeros(shape=(3*par.NimpOrbs))
        point[0:par.NimpOrbs] = epsTilde[:,0]
        point[par.NimpOrbs:2*par.NimpOrbs] = vTilde[:,0]
        point[2*par.NimpOrbs:3*par.NimpOrbs] = epsBath[:,0]
    else:
        point = np.zeros(shape=(3*par.numDeg))
        for iD in range(par.numDeg):
            point[iD] = epsTilde[par.degeneracy[iD][0],0]
            point[iD+par.numDeg] = vTilde[par.degeneracy[iD][0],0]
            point[iD+2*par.numDeg] = epsBath[par.degeneracy[iD][0],0]     
        
    return point
    
def paramsToVarVectorwU(par,epsTilde,vTilde,epsBath,U):
    #%%    
    if par.numDeg == 0:
        point = np.zeros(shape=(3*par.NimpOrbs+1))
        point[0:par.NimpOrbs] = epsTilde[:,0]
        point[par.NimpOrbs:2*par.NimpOrbs] = vTilde[:,0]
        point[2*par.NimpOrbs:3*par.NimpOrbs] = epsBath[:,0]
        point[3*par.NimpOrbs] = U
    else:
        point = np.zeros(shape=(3*par.numDeg+1))
        for iD in range(par.numDeg):
            point[iD] = epsTilde[par.degeneracy[iD][0],0]
            point[iD+par.numDeg] = vTilde[par.degeneracy[iD][0],0]
            point[iD+2*par.numDeg] = epsBath[par.degeneracy[iD][0],0]   
            point[3*par.numDeg] = U
        
    return point
    


def pointToVecOptAllLegendre(cfg,par,point):
    
    coeffsD = np.zeros(shape=(par.NimpOrbs,(1+par.Nbath)))
    coeffsC1 = np.zeros(shape=(par.NimpOrbs,(1+par.Nbath)))
    for iD in range(par.numDeg):
        for iOD in range(len(par.degeneracy[iD])):
            coeffsD[par.degeneracy[iD][iOD],0] = point[iD*(2+4*cfg['algo']['legendreOrder'])]
            cD = point[iD*(2+4*cfg['algo']['legendreOrder'])+1:iD*(2+4*cfg['algo']['legendreOrder'])+1+2*cfg['algo']['legendreOrder']]
            coeffsD[par.degeneracy[iD][iOD],1:] = fitFuncWrapNew(par.mu,cfg)(par.epsBath[par.degeneracy[iD][iOD],:],cD)
            coeffsC1[par.degeneracy[iD][iOD],0] = point[iD*(2+4*cfg['algo']['legendreOrder'])+1+2*cfg['algo']['legendreOrder']]
            cC1 = point[iD*(2+4*cfg['algo']['legendreOrder'])+2+2*cfg['algo']['legendreOrder']:iD*(2+4*cfg['algo']['legendreOrder'])+2+4*cfg['algo']['legendreOrder']]
            coeffsC1[par.degeneracy[iD][iOD],1:] = fitFuncWrapNew(par.mu,cfg)(par.epsBath[par.degeneracy[iD][iOD],:],cC1)
            
    return coeffsD, coeffsC1

def pointToVecOptAllSimple(cfg,par,point):
    coeffsD = np.zeros(shape=(par.NimpOrbs,(1+par.Nbath)))
    coeffsC1 = np.zeros(shape=(par.NimpOrbs,(1+par.Nbath)))
    
    coeffsDDeg = np.reshape(point[:(par.numDeg*(1+par.Nbath))],(par.numDeg,(1+par.Nbath)))
    coeffsC1Deg = np.reshape(point[(par.numDeg*(1+par.Nbath)):],(par.numDeg,(1+par.Nbath)))

    for iD in range(par.numDeg):
        for iOD in range(len(par.degeneracy[iD])):
            coeffsD[par.degeneracy[iD][iOD],:] = coeffsDDeg[iD,:]
            coeffsC1[par.degeneracy[iD][iOD],:] = coeffsC1Deg[iD,:]
    return coeffsD, coeffsC1
    

def pointToVecOnlyBathLegendre(cfg,par,point):
    coeffsD = np.zeros(shape=(par.NimpOrbs,(1+par.Nbath)))    
    coeffsD[:,0] = 1.0
    coeffsC1 = np.zeros(shape=(par.NimpOrbs,(1+par.Nbath)))
    for iD in range(par.numDeg):
        for iOD in range(len(par.degeneracy[iD])):
            cC1 = point[iD*(2*cfg['algo']['legendreOrder']):(iD+1)*(2*cfg['algo']['legendreOrder'])]
            coeffsC1[par.degeneracy[iD][iOD],1:] = fitFuncWrapNew(par.mu,cfg)(par.epsBath[par.degeneracy[iD][iOD],:],cC1)
    return coeffsD,coeffsC1
    
def pointToVecOnlyBathSimple(cfg,par,point):
    coeffsD = np.zeros(shape=(par.NimpOrbs,(1+par.Nbath)))    
    coeffsD[:,0] = 1.0
    coeffsC1 = np.zeros(shape=(par.NimpOrbs,(1+par.Nbath)))
    coeffsC1Deg = np.reshape(point,(par.numDeg,(par.Nbath)))
    for iD in range(par.numDeg):
        for iOD in range(len(par.degeneracy[iD])):
            coeffsC1[par.degeneracy[iD][iOD],1:] = coeffsC1Deg[iD,:]
    return coeffsD,coeffsC1
    
  
def pointToVecOrbStitch(cfg,par,point,orb,restPoint):
    #%% 

    # stitch the point into the restPoint
    combinedPoint = restPoint.copy()
    if par.numDeg == 0:
        lengthPoint = combinedPoint.size // par.NimpOrbs
    else:
        lengthPoint = combinedPoint.size // par.numDeg
    combinedPoint[orb*lengthPoint : (orb+1)*lengthPoint] = point
    
    #calculate the coeffs for the restPoint
    coeffsD, coeffsC1 = pointToVec(cfg,par,combinedPoint)
    
    return coeffsD,coeffsC1
  
def solveUncorrelatedProblem(par,onePartHamilUnCorrSubOrb):
    #%% 
    nDimOrb = onePartHamilUnCorrSubOrb.shape[1]
    nDim = onePartHamilUnCorrSubOrb.shape[1]*par.NimpOrbs
    
    unCorrEigVals = np.zeros(shape=nDim)
    nF = np.zeros(shape=nDim)
    #unCorrEigVecs = np.zeros(shape=(nDim,nDim))
    unCorrEigValsOrb = np.zeros(shape=(par.NimpOrbs,nDimOrb))
    unCorrEigVecsOrb = np.zeros(shape=(par.NimpOrbs,nDimOrb,nDimOrb))
    nFOrb = np.zeros(shape=(par.NimpOrbs,nDimOrb))
    for iD in range(par.numDeg):
        dumVals,dumVecs = np.linalg.eigh(onePartHamilUnCorrSubOrb[par.degeneracy[iD][0],:,:])
        nFDum = fermiFunc(par,dumVals)
        for iOD in range(len(par.degeneracy[iD])):
            unCorrEigValsOrb[par.degeneracy[iD][iOD],:] = dumVals
            nFOrb[par.degeneracy[iD][iOD],:] = nFDum
            unCorrEigVecsOrb[par.degeneracy[iD][iOD],:,:] = dumVecs

            unCorrEigVals[par.degeneracy[iD][iOD]*nDimOrb:(par.degeneracy[iD][iOD]+1)*nDimOrb] = dumVals
            nF[par.degeneracy[iD][iOD]*nDimOrb:(par.degeneracy[iD][iOD]+1)*nDimOrb] = nFDum
            #unCorrEigVecs[par.degeneracy[iD][iOD]*nDimOrb:(par.degeneracy[iD][iOD]+1)*nDimOrb,par.degeneracy[iD][iOD]*nDimOrb:(par.degeneracy[iD][iOD]+1)*nDimOrb] = dumVecs
    #np.savetxt('eigvals.dat',unCorrEigValsOrb)
                
    #nF = fermiFunc(par,unCorrEigVals)
    
    if par.onlyGroundState:
        PhiUnCorr =  2*np.sum(unCorrEigVals *nF)
        energyEffUnCorr = PhiUnCorr
    else:
        PhiUnCorr = -2/par.beta * np.sum(np.log(1+np.exp(-par.beta * (unCorrEigVals - par.mu))))        
        # if temperature is too small, this will likely result in inf
        # then simply use the groundstate energy as the free energy.
        if np.isinf(PhiUnCorr):
            PhiUnCorr =  2*np.sum(unCorrEigVals *nF)
        energyEffUnCorr = 2.0 * np.sum(unCorrEigVals * nF)

    
    #unCorrDenMat = 0.0
    # orbitally resolved:
    unCorrDenMatOrb = np.zeros(shape=(par.NimpOrbs,nDimOrb,nDimOrb))

    for iD in range(par.numDeg):
        #dummy = np.dot(unCorrEigVecsOrb[par.degeneracy[iD][0],:,:],np.dot(np.diag( nFOrb[par.degeneracy[iD][0],:]  ),unCorrEigVecsOrb[par.degeneracy[iD][0],:,:].transpose()))
        # this is faster:
        dummy = np.dot(unCorrEigVecsOrb[par.degeneracy[iD][0],:,:],(nFOrb[par.degeneracy[iD][0],:][np.newaxis,:]*unCorrEigVecsOrb[par.degeneracy[iD][0],:,:]).transpose())
        for iOD in range(len(par.degeneracy[iD])):
            unCorrDenMatOrb[par.degeneracy[iD][iOD],:,:] = dummy
    
    

    # kill unCorrEigVals,unCorrEigVecs,unCorrDenMat,
    return unCorrDenMatOrb, PhiUnCorr, energyEffUnCorr

def calcUpdatedObservables(pointX,*args):
    #%% 
    hami = args[0]
    cfg = args[1]
    par = args[2]
    parEff = args[3]
    blockCar = args[4]
    hFSol = args[5]
    
    epsTilde, vBar, epsBath = varVectorToParams(pointX,cfg,par)
    #epsTilde, vBar, epsBath, U = varVectorToParamswU(pointX,par)
    #parEff.UImp = U
    #parEff.JImp = U/4.0
    #parEff.uMatrix = uMat.uMatrixWrapper(parEff)
    parEff.epsImp = epsTilde[:,0]
    parEff.vBath[:,0] = vBar[:,0]
    parEff.epsBath[:,0] = epsBath[:,0]
    
    #edSol = funcEDPhi(par,parEff,spPars,blockCar,hami)
    edSol = io.fetchLastEDSolution(cfg)
    #corrDenMat, Phi, E0, twoPart, energyCorr = funcEDPhi(par,parEff,spPars,blockCar,hami)
        
    # set up uncorrelated rest Problem:
    if os.path.isfile(cfg['tempFileConstrain']):
        _, coeffsD, coeffsC1 = io.getRecentOnePartBasis(cfg)
    else:
        raise Exception('no opt. one particle basis found!')
    for iO in range(par.NimpOrbs):
        coeffsC1[iO,:] /= np.sqrt(np.sum(coeffsC1[iO,:]**2))
        coeffsD[iO,:] /= np.sqrt(np.sum(coeffsD[iO,:]**2))
    onePartHamilUnCorrSub , onePartHamilUnCorrSubOrb ,coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(cfg,par,hFSol,coeffsD,coeffsC1)
    
    # solve the uncorrelated Problem orbitlly blockwise:
    _,_,_,unCorrDenMatOrb,PhiUnCorr, energyEffUnCorr = solveUncorrelatedProblem(par,onePartHamilUnCorrSubOrb)

    energyOne,energyTwo,energy, completeDenMatLocBasis,twoPartLocUpDn, twoPartLocUpUp = stitchEnergy(cfg, par,edSol['corrDenMat'],edSol['twoPart'],unCorrDenMatOrb,coeffsComplete,coeffsOrbComplete)
    energyEff = energyEffUnCorr + edSol['energyEffCorr']
    PhiComplete = edSol['PhiCorr'] + PhiUnCorr
    print('\nDetailed energy and free energy')
    print('phi* C:      {:+1.6f}'.format(edSol['PhiCorr']))
    print('phi* C\':     {:+1.6f}'.format(PhiUnCorr))
    print('phi* C+C\':   {:+1.6f}'.format( PhiComplete))
    print('<H*>* C:     {:+1.6f}'.format(edSol['energyEffCorr']))
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
    for iO in range(par.NimpOrbs):
        SzSquare += 0.5*completeDenMatLocBasis[iO,iO]
        for jO in range(par.NimpOrbs):
            SzSquare += -0.5* twoPartLocUpUp[iO,jO,iO,jO]- 0.5*twoPartLocUpDn[iO,jO,jO,iO]
    # <S+S- + S-S+>
    SpmmpSquare = 2.0 * np.trace(completeDenMatLocBasis[:par.NimpOrbs,:par.NimpOrbs])
    for iO in range(par.NimpOrbs):
        for jO in range(par.NimpOrbs):
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

    

def calcOriginalSpectraBetaMatrixElemMinus(par,ham,cfg,eigVals,eigVecs,alpha,beta):

    w = np.linspace(cfg['ed']['minW'],cfg['ed']['maxW'],cfg['ed']['numW'])
    
    if cfg['ed']['calcSpectra'] == False:
       print('not calculating spectral function!',flush=True)
       dos = np.zeros(shape=w.shape)
    else:
        if cfg['aim']['onlyGroundState'] == True:
            raise Exception('to calculate the spectral functions, do thermodynamic calculation')
        print('calculating spectral function... ',flush=True)
        
        # the thermodynamic cut off is of no use for calculating the spectral function
        
 
        # recalcuate the partition function without cut off
        ZZ1Til,ZZ1Block,E0,minIndex,E0Index,thermo = partitionFunctionSpectra(par,ham,eigVals)
        numOnePartStates = par.NimpOrbs + par.NimpOrbs*par.Nbath
        
        # store the matrices in short variable names
        anni = []
        for ii in range(numOnePartStates):
            anni.append(ham['oper']['-'+str(ii)+'_up'])
        
        # store the quantum numbers list in an array
        quantumNums = np.zeros(shape=(len(ham['diagblk_qnum']),len(ham['diagblk_qnum'][0])),dtype=int)
    
        for ii in range(len(ham['diagblk_qnum'])):
            quantumNums[ii,0] = ham['diagblk_qnum'][ii][0]
            quantumNums[ii,1] = ham['diagblk_qnum'][ii][1]
            quantumNums[ii,2] = ham['diagblk_qnum'][ii][2]
        
        # store the spectrum in here        
        spec = np.zeros(shape=(w.size),dtype=complex)
        # loop over all degenerate ground state states

        for iBlock in range(len(eigVals)):
            if thermo[iBlock] > 0.0:
                print('\n block',quantumNums[iBlock,0],'importance',thermo[iBlock])

            #print(ZZ1Block[iBlock])
            
            for iVal in range(eigVals[iBlock].size):
                
                nVec = eigVecs[iBlock][:,iVal]
                En = eigVals[iBlock][iVal]
                tempN = ZZ1Block[iBlock][iVal]
                if tempN < cfg['ed']['betaCutOffExpecValue']:
                    continue
                if iVal%10 == 0:
                    print(iVal,end=' ',flush=True)
                
                booleanPartNumberAnni = quantumNums[:,0] == -1+ ham['diagblk_qnum'][iBlock][0]
                booleanSzDnNumberAnni = quantumNums[:,1] ==  ham['diagblk_qnum'][iBlock][1]
                booleanSzUpNumberAnni = quantumNums[:,2] == -1+ ham['diagblk_qnum'][iBlock][2]
                # all quantum numbers have to fit
                booleanAnni = (np.vstack((booleanPartNumberAnni,booleanSzUpNumberAnni,booleanSzDnNumberAnni)).all(axis=0))
                
                mdnAlpha = []
                mdnBeta = []
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    mVecs = eigVecs[jBlock]
                    matBlockAlpha = anni[alpha][ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBeta = anni[beta][ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    mdnAlpha.append(np.sum(matBlockAlpha.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
                    mdnBeta.append(np.sum(matBlockBeta.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
            
                count = 0
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    Em =eigVals[jBlock]
                    spec += tempN*np.sum((mdnAlpha[count]*mdnBeta[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (-En + Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1                            
    return w, 1.0/np.pi*spec.imag/ZZ1Til
    

def calcOriginalSpectraBetaMatrixElem(par,ham,cfg,eigVals,eigVecs,alpha,beta):

    w = np.linspace(cfg['ed']['minW'],cfg['ed']['maxW'],cfg['ed']['numW'])
    
    if cfg['ed']['calcSpectra'] == False:
       print('not calculating spectral function!',flush=True)
       dos = np.zeros(shape=w.shape)
    else:
        if cfg['aim']['onlyGroundState'] == True:
            raise Exception('to calculate the spectral functions, do thermodynamic calculation')
        print('calculating spectral function... ',flush=True)
        
        # the thermodynamic cut off is of no use for calculating the spectral function
        
 
        # recalcuate the partition function without cut off
        ZZ1Til,ZZ1Block,E0,minIndex,E0Index,thermo = partitionFunctionSpectra(par,ham,eigVals)
        numOnePartStates = par.NimpOrbs + par.NimpOrbs*par.Nbath
        
        # store the matrices in short variable names
        crea = []
        anni = []
        for ii in range(numOnePartStates):
            crea.append(ham['oper']['+'+str(ii)+'_up'])
            anni.append(ham['oper']['-'+str(ii)+'_up'])
        
        # store the quantum numbers list in an array
        quantumNums = np.zeros(shape=(len(ham['diagblk_qnum']),len(ham['diagblk_qnum'][0])),dtype=int)
    
        for ii in range(len(ham['diagblk_qnum'])):
            quantumNums[ii,0] = ham['diagblk_qnum'][ii][0]
            quantumNums[ii,1] = ham['diagblk_qnum'][ii][1]
            quantumNums[ii,2] = ham['diagblk_qnum'][ii][2]
        
        # store the spectrum in here        
        spec = np.zeros(shape=(w.size),dtype=complex)
        # loop over all degenerate ground state states

        for iBlock in range(len(eigVals)):
            if thermo[iBlock] > 0.0:
                print('\n block',quantumNums[iBlock,0],'importance',thermo[iBlock])

            #print(ZZ1Block[iBlock])
            
            for iVal in range(eigVals[iBlock].size):
                
                nVec = eigVecs[iBlock][:,iVal]
                En = eigVals[iBlock][iVal]
                tempN = ZZ1Block[iBlock][iVal]
                if tempN < cfg['ed']['betaCutOffExpecValue']:
                    continue
                if iVal%10 == 0:
                    print(iVal,end=' ',flush=True)
                # find the block in which we will land:
                booleanPartNumberCrea = quantumNums[:,0] == 1+ ham['diagblk_qnum'][iBlock][0]
                booleanSzDnNumberCrea = quantumNums[:,1] ==  ham['diagblk_qnum'][iBlock][1]
                booleanSzUpNumberCrea = quantumNums[:,2] == 1+ ham['diagblk_qnum'][iBlock][2]
                # all quantum numbers have to fit
                booleanCrea = (np.vstack((booleanPartNumberCrea,booleanSzUpNumberCrea,booleanSzDnNumberCrea)).all(axis=0))
                end= 0.0
                start = time.time()
                mdnAlpha = []
                mdnBeta = []
                for jBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
                    mVecs = eigVecs[jBlock]
                    matBlockAlpha = crea[alpha][ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBeta = crea[beta][ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    
                    mdnAlpha.append(np.sum(matBlockAlpha.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
                    mdnBeta.append(np.sum(matBlockBeta.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
                count = 0
                for jBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
                    Em =eigVals[jBlock]
                    spec += tempN*np.sum((mdnAlpha[count]*mdnBeta[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1
                booleanPartNumberAnni = quantumNums[:,0] == -1+ ham['diagblk_qnum'][iBlock][0]
                booleanSzDnNumberAnni = quantumNums[:,1] ==  ham['diagblk_qnum'][iBlock][1]
                booleanSzUpNumberAnni = quantumNums[:,2] == -1+ ham['diagblk_qnum'][iBlock][2]
                # all quantum numbers have to fit
                booleanAnni = (np.vstack((booleanPartNumberAnni,booleanSzUpNumberAnni,booleanSzDnNumberAnni)).all(axis=0))
                
                mdnAlpha = []
                mdnBeta = []
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    mVecs = eigVecs[jBlock]
                    matBlockAlpha = anni[alpha][ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBeta = anni[beta][ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    mdnAlpha.append(np.sum(matBlockAlpha.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
                    mdnBeta.append(np.sum(matBlockBeta.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
            
                count = 0
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    Em =eigVals[jBlock]
                    spec += tempN*np.sum((mdnAlpha[count]*mdnBeta[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (-En + Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1                            
    return w, 1.0/np.pi*spec.imag/ZZ1Til
    
    
def calcOriginalSpectraBetaMatrix(par,ham,cfg,eigVals,eigVecs):

    numOnePartStates = par.NimpOrbs + par.NimpOrbs*par.Nbath
    
    dos = np.zeros(shape=(cfg['ed']['numW'],numOnePartStates,numOnePartStates))
    for alpha in range(numOnePartStates):
        for beta in range(numOnePartStates):
            print(alpha,beta)
            if beta >= alpha:
                w, dos[:,alpha,beta] = calcOriginalSpectraBetaMatrixElem(par,ham,cfg,eigVals,eigVecs,alpha,beta)
            else:
                dos[:,alpha,beta] = dos[:,beta,alpha]
            
    return w, dos
    
def calcOriginalSpectraBetaMatrixMinus(par,ham,cfg,eigVals,eigVecs):

    numOnePartStates = par.NimpOrbs + par.NimpOrbs*par.Nbath
    epsdMatrix=  np.diag(par.epsImp)
    hamNonInt=harFock.onePartMatrix(par.epsBath,par.vBath,epsdMatrix,par.Nbath)
    dos = np.zeros(shape=(cfg['ed']['numW'],numOnePartStates,numOnePartStates))
    for alpha in range(numOnePartStates):
        for beta in range(numOnePartStates):
            if beta >= alpha:
                print(alpha,beta)
                if (hamNonInt[alpha,beta] != 0.0) or (alpha == beta):
                    w, dos[:,alpha,beta] = calcOriginalSpectraBetaMatrixElemMinus(par,ham,cfg,eigVals,eigVecs,alpha,beta)
            else:
                dos[:,alpha,beta] = dos[:,beta,alpha]
            
    return w, dos
    
def calcOriginalSpectraBeta(par,ham,cfg,eigVals,eigVecs):
    
    w, dos = calcOriginalSpectraBetaMatrixElem(par,ham,cfg,eigVals,eigVecs,0,0)
    
    return w, dos

def calcOriginalSpectra(par,ham,cfg,eigVals,eigVecs):
    
    
    
    w = np.linspace(cfg['ed']['minW'],cfg['ed']['maxW'],cfg['ed']['numW'])
    
    if cfg['ed']['calcSpectra'] == False:
       print('not calculating spectral function!',flush=True)
       dos = np.zeros(shape=w.shape)
    else:
        if cfg['aim']['onlyGroundState'] == True:
            raise Exception('to calculate the spectral functions, do thermodynamic calculation')
        print('calculating spectral function... ',flush=True)
        
        # the thermodynamic cut off is of no use for calculating the spectral function
        
 
        # recalcuate the partition function without cut off
        ZZ1Til,ZZ1Block,E0,minIndex,E0Index,thermo = partitionFunctionSpectra(par,ham,eigVals)


        # store the matrices in short variable names
        crea = ham['oper']['+imp1_up']
        anni = ham['oper']['-imp1_up']
        
        # store the quantum numbers list in an array
        quantumNums = np.zeros(shape=(len(ham['diagblk_qnum']),len(ham['diagblk_qnum'][0])),dtype=int)
    
        for ii in range(len(ham['diagblk_qnum'])):
            quantumNums[ii,0] = ham['diagblk_qnum'][ii][0]
            quantumNums[ii,1] = ham['diagblk_qnum'][ii][1]
            quantumNums[ii,2] = ham['diagblk_qnum'][ii][2]
        
        # store the spectrum in here        
        spec = np.zeros(shape=(w.size),dtype=complex)
        # loop over all degenerate ground state states
        for iDeg in range(len(minIndex)):
            En = E0[iDeg]
            for jDeg in range(len(E0Index[iDeg])):
                nVec = eigVecs[minIndex[iDeg]][:,E0Index[iDeg][jDeg]]
                
                # find the block in which we will land:
                booleanPartNumberCrea = quantumNums[:,0] == 1+ ham['diagblk_qnum'][minIndex[iDeg]][0]
                booleanSzDnNumberCrea = quantumNums[:,1] ==  ham['diagblk_qnum'][minIndex[iDeg]][1]
                booleanSzUpNumberCrea = quantumNums[:,2] == 1+ ham['diagblk_qnum'][minIndex[iDeg]][2]
                # all quantum numbers have to fit
                booleanCrea = (np.vstack((booleanPartNumberCrea,booleanSzUpNumberCrea,booleanSzDnNumberCrea)).all(axis=0))
                
                mdn = []
                for iBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
            
                    mVecs = eigVecs[iBlock]
                    matBlockD = crea[ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1],ham['diagblk_ind'][minIndex[iDeg]]:ham['diagblk_ind'][minIndex[iDeg]+1]]
                    
                    mdn.append(np.sum(matBlockD.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
                
                count = 0
                for iBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
                    Em =eigVals[iBlock]
                    spec += np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1
                booleanPartNumberAnni = quantumNums[:,0] == -1+ ham['diagblk_qnum'][minIndex[iDeg]][0]
                booleanSzDnNumberAnni = quantumNums[:,1] ==  ham['diagblk_qnum'][minIndex[iDeg]][1]
                booleanSzUpNumberAnni = quantumNums[:,2] == -1+ ham['diagblk_qnum'][minIndex[iDeg]][2]
                # all quantum numbers have to fit
                booleanAnni = (np.vstack((booleanPartNumberAnni,booleanSzUpNumberAnni,booleanSzDnNumberAnni)).all(axis=0))
                
                mdn = []
                for iBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    mVecs = eigVecs[iBlock]
                    matBlockD = anni[ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1],ham['diagblk_ind'][minIndex[iDeg]]:ham['diagblk_ind'][minIndex[iDeg]+1]]
                    mdn.append(np.sum(matBlockD.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
            
                count = 0
                for iBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    Em =eigVals[iBlock]
                    spec += np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (-En + Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1                              
        dos = 1.0/np.pi*spec.imag/ZZ1Til
    return w, dos

def calcEffectiveSpectra(par,ham,cfg,hFSol):
    
    edSol = io.fetchLastEDSolution(cfg)
    eigVals = edSol['eigValsFull']
    eigVecs = edSol['eigVecsFull']
    
    # set up uncorrelated rest Problem:
    print(cfg['tempFileConstrain'])
    if os.path.isfile(cfg['tempFileConstrain']):
        _, coeffsD, coeffsC1 = io.getRecentOnePartBasis(cfg)
    else:
        raise Exception('no opt. one particle basis found!')
    for iO in range(par.NimpOrbs):
        coeffsC1[iO,:] /= np.sqrt(np.sum(coeffsC1[iO,:]**2))
        coeffsD[iO,:] /= np.sqrt(np.sum(coeffsD[iO,:]**2))
    onePartHamilUnCorrSub,onePartHamilUnCorrSubOrb, coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(cfg,par,hFSol,coeffsD,coeffsC1)

    # solve the uncorrelated Problem orbitlly blockwise:
    unCorrEigVals,_,_,_,_, energyEffUnCorr = solveUncorrelatedProblem(par,onePartHamilUnCorrSubOrb)
    
    unCorrDenMat = np.diag((-np.sign(unCorrEigVals - par.mu)+1)*0.5)
    
    w = np.linspace(cfg['ed']['minW'],cfg['ed']['maxW'],cfg['ed']['numW'])
    
    if cfg['ed']['calcSpectra'] == False:
       print('not calculating spectral function!',flush=True)
       dos = np.zeros(shape=w.shape)
    else:
        if cfg['aim']['onlyGroundState'] == True:
            raise Exception('to calculate the spectral functions, do thermodynamic calculation')
        print('calculating spectral function... ',flush=True)
        
        # the thermodynamic cut off is of no use for calculating the spectral function
        
      
        # recalcuate the partition function without cut off
        ZZ1Til,ZZ1Block,E0,minIndex,E0Index,thermo = partitionFunctionSpectra(par,ham,eigVals)
        


        # store the matrices in short variable names
        creaImp = ham['oper']['+imp1_up']
        anniImp = ham['oper']['-imp1_up']
        creaBath = ham['oper']['+bath1_1_up']
        anniBath = ham['oper']['-bath1_1_up']
        
        # store the quantum numbers list in an array
        quantumNums = np.zeros(shape=(len(ham['diagblk_qnum']),len(ham['diagblk_qnum'][0])),dtype=int)
    
        for ii in range(len(ham['diagblk_qnum'])):
            quantumNums[ii,0] = ham['diagblk_qnum'][ii][0]
            quantumNums[ii,1] = ham['diagblk_qnum'][ii][1]
            quantumNums[ii,2] = ham['diagblk_qnum'][ii][2]
        
        # store the spectrum in here        
        spec = np.zeros(shape=(w.size),dtype=complex)
        print('')
        print('noninteracting peaks at')
        print(unCorrEigVals)
        print('weighted with')
        print(coeffsComplete[0,2*par.NimpOrbs:])
        for ik in range(unCorrEigVals.size):
            mcknT = coeffsComplete[0,2*par.NimpOrbs+ik]
            spec += (mcknT*mcknT.conjugate()) / (w + (-unCorrEigVals[ik] - 1j*cfg['ed']['idelta']))

        # loop over all degenerate ground state states
        for iDeg in range(len(minIndex)):
            En = E0[iDeg]
            for jDeg in range(len(E0Index[iDeg])):
                nVec = eigVecs[minIndex[iDeg]][:,E0Index[iDeg][jDeg]]
                
                # find the block in which we will land:
                booleanPartNumberCrea = quantumNums[:,0] == 1+ ham['diagblk_qnum'][minIndex[iDeg]][0]
                booleanSzDnNumberCrea = quantumNums[:,1] ==  ham['diagblk_qnum'][minIndex[iDeg]][1]
                booleanSzUpNumberCrea = quantumNums[:,2] == 1+ ham['diagblk_qnum'][minIndex[iDeg]][2]
                # all quantum numbers have to fit
                booleanCrea = (np.vstack((booleanPartNumberCrea,booleanSzUpNumberCrea,booleanSzDnNumberCrea)).all(axis=0))
                
                mdn = []
                for iBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
            
                    mVecs = eigVecs[iBlock]
                    matBlockImp = creaImp[ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1],ham['diagblk_ind'][minIndex[iDeg]]:ham['diagblk_ind'][minIndex[iDeg]+1]]
                    matBlockBath = creaBath[ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1],ham['diagblk_ind'][minIndex[iDeg]]:ham['diagblk_ind'][minIndex[iDeg]+1]]
                    
                    mdnImp = np.sum(matBlockImp.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnBath = np.sum(matBlockBath.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdn.append(coeffsD[0,0]*mdnImp + coeffsC1[0,0]*mdnBath)
                
                count = 0
                for iBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
                    Em =eigVals[iBlock]
                    spec += np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1
                booleanPartNumberAnni = quantumNums[:,0] == -1+ ham['diagblk_qnum'][minIndex[iDeg]][0]
                booleanSzDnNumberAnni = quantumNums[:,1] ==  ham['diagblk_qnum'][minIndex[iDeg]][1]
                booleanSzUpNumberAnni = quantumNums[:,2] == -1+ ham['diagblk_qnum'][minIndex[iDeg]][2]
                # all quantum numbers have to fit
                booleanAnni = (np.vstack((booleanPartNumberAnni,booleanSzUpNumberAnni,booleanSzDnNumberAnni)).all(axis=0))
                
                mdn = []
                for iBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    mVecs = eigVecs[iBlock]
                    matBlockImp = anniImp[ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1],ham['diagblk_ind'][minIndex[iDeg]]:ham['diagblk_ind'][minIndex[iDeg]+1]]
                    matBlockBath = anniBath[ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1],ham['diagblk_ind'][minIndex[iDeg]]:ham['diagblk_ind'][minIndex[iDeg]+1]]
                    
                    mdnImp = np.sum(matBlockImp.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnBath = np.sum(matBlockBath.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    
                    mdn.append(coeffsD[0,0]*mdnImp + coeffsC1[0,0]*mdnBath)
            
                count = 0
                for iBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    Em =eigVals[iBlock]
                    spec += np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (-En + Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1                              
        dos = 1.0/np.pi*spec.imag/ZZ1Til
    return w, dos

def calcEffectiveSpectraBetaMatrixElem(par,ham,cfg,hFSol,alpha,beta):
    
    edSol = io.fetchLastEDSolution(cfg)
    eigVals = edSol['eigValsFull']
    eigVecs = edSol['eigVecsFull']
    
    # set up uncorrelated rest Problem:
    print(cfg['tempFileConstrain'])
    if os.path.isfile(cfg['tempFileConstrain']):
        _, coeffsD, coeffsC1 = io.getRecentOnePartBasis(cfg)
    else:
        raise Exception('no opt. one particle basis found!')
    for iO in range(par.NimpOrbs):
        coeffsC1[iO,:] /= np.sqrt(np.sum(coeffsC1[iO,:]**2))
        coeffsD[iO,:] /= np.sqrt(np.sum(coeffsD[iO,:]**2))
    onePartHamilUnCorrSub,onePartHamilUnCorrSubOrb, coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(cfg,par,hFSol,coeffsD,coeffsC1)

    # solve the uncorrelated Problem orbitlly blockwise:
    unCorrEigVals,unCorrEigVecs,_,_,_, energyEffUnCorr = solveUncorrelatedProblem(par,onePartHamilUnCorrSubOrb)

    unCorrDenMat = np.diag(fermiFunc(par,unCorrEigVals))
    
    w = np.linspace(cfg['ed']['minW'],cfg['ed']['maxW'],cfg['ed']['numW'])
    
    if cfg['ed']['calcSpectra'] == False:
       print('not calculating spectral function!',flush=True)
       dos = np.zeros(shape=w.shape)
    else:
        if cfg['aim']['onlyGroundState'] == True:
            raise Exception('to calculate the spectral functions, do thermodynamic calculation')
        print('calculating spectral function... ',flush=True)
        
        # the thermodynamic cut off is of no use for calculating the spectral function
        
      
        # recalcuate the partition function without cut off
        ZZ1Til,ZZ1Block,E0,minIndex,E0Index,thermo = partitionFunctionSpectra(par,ham,eigVals)
        
        # store the matrices in short variable names
        creaImp = ham['oper']['+imp1_up']
        anniImp = ham['oper']['-imp1_up']
        creaBath = ham['oper']['+bath1_1_up']
        anniBath = ham['oper']['-bath1_1_up']
        
        # store the quantum numbers list in an array
        quantumNums = np.zeros(shape=(len(ham['diagblk_qnum']),len(ham['diagblk_qnum'][0])),dtype=int)
    
        for ii in range(len(ham['diagblk_qnum'])):
            quantumNums[ii,0] = ham['diagblk_qnum'][ii][0]
            quantumNums[ii,1] = ham['diagblk_qnum'][ii][1]
            quantumNums[ii,2] = ham['diagblk_qnum'][ii][2]
        
        # store the spectrum in here
        numOnePartStates = par.NimpOrbs + par.NimpOrbs*par.Nbath

        spec = np.zeros(shape=(w.size),dtype=complex)

        #
        # uncorrelated part of the spectrum:
        #

        # transformation from eigenbasis of the uncorrelated basis to
        # first guess, orthogonal basis in which hamitlonian is projected
        # and then to the original basis
        uVAllAlpha = np.dot(coeffsOrbComplete[0,alpha,2*par.NimpOrbs:],unCorrEigVecs)
        uVAllBeta = np.dot(coeffsOrbComplete[0,beta,2*par.NimpOrbs:],unCorrEigVecs)

        # sum over all eigenenergies of the uncorrelated system
        # weigthed with the overlap to the d state
        for ik in range(unCorrEigVals.size):
            spec += (uVAllAlpha[ik]*uVAllBeta[ik].conjugate()) / (w + (-unCorrEigVals[ik] - 1j*cfg['ed']['idelta']))

        #
        # correlated part of the spectrum
        #

        # loop over all degenerate ground state states
        for iBlock in range(len(eigVals)):
            if thermo[iBlock] > 0.0:
                print('\n block',quantumNums[iBlock,0],'importance',thermo[iBlock])
            for iVal in range(eigVals[iBlock].size):
                nVec = eigVecs[iBlock][:,iVal]
                En = eigVals[iBlock][iVal]
                tempN = ZZ1Block[iBlock][iVal]
                if tempN < cfg['ed']['betaCutOffExpecValue']:
                    continue
                
                # find the block in which we will land:
                booleanPartNumberCrea = quantumNums[:,0] == 1+ ham['diagblk_qnum'][iBlock][0]
                booleanSzDnNumberCrea = quantumNums[:,1] ==  ham['diagblk_qnum'][iBlock][1]
                booleanSzUpNumberCrea = quantumNums[:,2] == 1+ ham['diagblk_qnum'][iBlock][2]
                # all quantum numbers have to fit
                booleanCrea = (np.vstack((booleanPartNumberCrea,booleanSzUpNumberCrea,booleanSzDnNumberCrea)).all(axis=0))
                
                mdnAlpha = []
                mdnBeta = []
                mdnSq = []
                for jBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
            
                    mVecs = eigVecs[jBlock]
                    matBlockImp = creaImp[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBath = creaBath[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    
                    mdnImp = np.sum(matBlockImp.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnBath = np.sum(matBlockBath.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnAlpha.append(coeffsD[0,alpha]*mdnImp + coeffsC1[0,alpha]*mdnBath)
                    mdnBeta.append(coeffsD[0,beta]*mdnImp + coeffsC1[0,beta]*mdnBath)
                
                count = 0
                for jBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
                    Em =eigVals[jBlock]
                    spec += tempN*np.sum((mdnAlpha[count]*mdnBeta[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1
                    
                booleanPartNumberAnni = quantumNums[:,0] == -1+ ham['diagblk_qnum'][iBlock][0]
                booleanSzDnNumberAnni = quantumNums[:,1] ==  ham['diagblk_qnum'][iBlock][1]
                booleanSzUpNumberAnni = quantumNums[:,2] == -1+ ham['diagblk_qnum'][iBlock][2]
                # all quantum numbers have to fit
                booleanAnni = (np.vstack((booleanPartNumberAnni,booleanSzUpNumberAnni,booleanSzDnNumberAnni)).all(axis=0))
                
                mdnAlpha = []
                mdnBeta = []
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    mVecs = eigVecs[jBlock]
                    matBlockImp = anniImp[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBath = anniBath[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    
                    mdnImp = np.sum(matBlockImp.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnBath = np.sum(matBlockBath.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    
                    mdnAlpha.append(coeffsD[0,alpha]*mdnImp + coeffsC1[0,alpha]*mdnBath)
                    mdnBeta.append(coeffsD[0,beta]*mdnImp + coeffsC1[0,beta]*mdnBath)
            
                count = 0
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    Em =eigVals[jBlock]
                    spec += tempN*np.sum((mdnAlpha[count]*mdnBeta[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (-En + Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1                               
        dos = 1.0/np.pi*spec.imag/ZZ1Til
    return w, dos
    

def calcEffectiveSpectraBetaMatrixElemMinus(par,ham,cfg,hFSol,alpha,beta):
    
    edSol = io.fetchLastEDSolution(cfg)
    eigVals = edSol['eigValsFull']
    eigVecs = edSol['eigVecsFull']
    
    # set up uncorrelated rest Problem:
    print(cfg['tempFileConstrain'])
    if os.path.isfile(cfg['tempFileConstrain']):
        _, coeffsD, coeffsC1 = io.getRecentOnePartBasis(cfg)
    else:
        raise Exception('no opt. one particle basis found!')
    for iO in range(par.NimpOrbs):
        coeffsC1[iO,:] /= np.sqrt(np.sum(coeffsC1[iO,:]**2))
        coeffsD[iO,:] /= np.sqrt(np.sum(coeffsD[iO,:]**2))
    onePartHamilUnCorrSub,onePartHamilUnCorrSubOrb, coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(cfg,par,hFSol,coeffsD,coeffsC1)

    # solve the uncorrelated Problem orbitlly blockwise:
    unCorrEigVals,unCorrEigVecs,_,_,_, energyEffUnCorr = solveUncorrelatedProblem(par,onePartHamilUnCorrSubOrb)

    unCorrDenMatDiag = np.diag(fermiFunc(par,unCorrEigVals))
    unCorrDenMat = np.dot(unCorrEigVecs,np.dot(unCorrDenMatDiag,unCorrEigVecs.transpose().conjugate()))
    w = np.linspace(cfg['ed']['minW'],cfg['ed']['maxW'],cfg['ed']['numW'])
    
    if cfg['ed']['calcSpectra'] == False:
       print('not calculating spectral function!',flush=True)
       dos = np.zeros(shape=w.shape)
    else:
        if cfg['aim']['onlyGroundState'] == True:
            raise Exception('to calculate the spectral functions, do thermodynamic calculation')
        print('calculating spectral function... ',flush=True)
        
        # the thermodynamic cut off is of no use for calculating the spectral function
        
      
        # recalcuate the partition function without cut off
        ZZ1Til,ZZ1Block,E0,minIndex,E0Index,thermo = partitionFunctionSpectra(par,ham,eigVals)
        
        # store the matrices in short variable names
        creaImp = ham['oper']['+imp1_up']
        anniImp = ham['oper']['-imp1_up']
        creaBath = ham['oper']['+bath1_1_up']
        anniBath = ham['oper']['-bath1_1_up']
        
        # store the quantum numbers list in an array
        quantumNums = np.zeros(shape=(len(ham['diagblk_qnum']),len(ham['diagblk_qnum'][0])),dtype=int)
    
        for ii in range(len(ham['diagblk_qnum'])):
            quantumNums[ii,0] = ham['diagblk_qnum'][ii][0]
            quantumNums[ii,1] = ham['diagblk_qnum'][ii][1]
            quantumNums[ii,2] = ham['diagblk_qnum'][ii][2]
        
        # store the spectrum in here
        numOnePartStates = par.NimpOrbs + par.NimpOrbs*par.Nbath

        spec = np.zeros(shape=(w.size),dtype=complex)

        #
        # uncorrelated part of the spectrum:
        #

        # transformation from eigenbasis of the uncorrelated basis to
        # first guess, orthogonal basis in which hamiltonian is projected
        # and then to the original basis
        uVAllAlpha = np.dot(coeffsOrbComplete[0,alpha,2*par.NimpOrbs:],unCorrEigVecs*fermiFunc(par,unCorrEigVals)[np.newaxis,:])
        uVAllBeta = np.dot(coeffsOrbComplete[0,beta,2*par.NimpOrbs:],unCorrEigVecs*fermiFunc(par,unCorrEigVals)[np.newaxis,:])
        #uVAllAlpha = np.dot(coeffsOrbComplete[0,alpha,2*par.NimpOrbs:],unCorrEigVecs)
        #uVAllBeta = np.dot(coeffsOrbComplete[0,beta,2*par.NimpOrbs:],unCorrEigVecs)
        #this = np.dot(coeffsOrbComplete[0,alpha,2*par.NimpOrbs:],np.dot(unCorrDenMat,coeffsOrbComplete[0,beta,2*par.NimpOrbs:].transpose().conjugate()))
        #print(this)
        # sum over all eigenenergies of the uncorrelated system
        # weigthed with the overlap to the d state
        for ik in range(unCorrEigVals.size):
            if alpha == beta:
                spec += (uVAllAlpha[ik]*uVAllBeta[ik].conjugate()) / (w + (-unCorrEigVals[ik] - 1j*cfg['ed']['idelta']))
            #print('this',this)
            #spec += (this[ik]) / (w + (-unCorrEigVals[ik] - 1j*cfg['ed']['idelta']))

        #
        # correlated part of the spectrum
        #

        # loop over all degenerate ground state states
        for iBlock in range(len(eigVals)):
            if thermo[iBlock] > 0.0:
                print('\n block',quantumNums[iBlock,0],'importance',thermo[iBlock])
            for iVal in range(eigVals[iBlock].size):
                nVec = eigVecs[iBlock][:,iVal]
                En = eigVals[iBlock][iVal]
                tempN = ZZ1Block[iBlock][iVal]
                if tempN < cfg['ed']['betaCutOffExpecValue']:
                    continue
                               
                booleanPartNumberAnni = quantumNums[:,0] == -1+ ham['diagblk_qnum'][iBlock][0]
                booleanSzDnNumberAnni = quantumNums[:,1] ==  ham['diagblk_qnum'][iBlock][1]
                booleanSzUpNumberAnni = quantumNums[:,2] == -1+ ham['diagblk_qnum'][iBlock][2]
                # all quantum numbers have to fit
                booleanAnni = (np.vstack((booleanPartNumberAnni,booleanSzUpNumberAnni,booleanSzDnNumberAnni)).all(axis=0))
                
                mdnAlpha = []
                mdnBeta = []
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    mVecs = eigVecs[jBlock]
                    matBlockImp = anniImp[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBath = anniBath[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    
                    mdnImp = np.sum(matBlockImp.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnBath = np.sum(matBlockBath.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    
                    mdnAlpha.append(coeffsD[0,alpha]*mdnImp + coeffsC1[0,alpha]*mdnBath)
                    mdnBeta.append(coeffsD[0,beta]*mdnImp + coeffsC1[0,beta]*mdnBath)
            
                count = 0
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    Em =eigVals[jBlock]
                    spec += tempN*np.sum((mdnAlpha[count]*mdnBeta[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (-En + Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1                               
        dos = 1.0/np.pi*spec.imag/ZZ1Til
    return w, dos

def calcEffectiveSpectraBetaMatrix(par,ham,cfg,hFSol):
    
    numOnePartStates = par.NimpOrbs + par.NimpOrbs*par.Nbath

    dos = np.zeros(shape=(cfg['ed']['numW'],numOnePartStates,numOnePartStates),dtype=complex)
    for alpha in range(numOnePartStates):
        for beta in range(numOnePartStates):
            w,dos[:,alpha,beta] = calcEffectiveSpectraBetaMatrixElem(par,ham,cfg,hFSol,alpha,beta)
    
    return w, dos
    
    
def calcEffectiveSpectraBetaMatrixMinus(par,ham,cfg,hFSol):
    
    numOnePartStates = par.NimpOrbs + par.NimpOrbs*par.Nbath
    epsdMatrix=  np.diag(par.epsImp)
    hamNonInt=harFock.onePartMatrix(par.epsBath,par.vBath,epsdMatrix,par.Nbath)

    dos = np.zeros(shape=(cfg['ed']['numW'],numOnePartStates,numOnePartStates),dtype=complex)
    for alpha in range(numOnePartStates):
        for beta in range(numOnePartStates):
            if beta >= alpha:
                if (hamNonInt[alpha,beta] != 0.0) or (alpha == beta):
                    print(alpha,beta)
                    w,dos[:,alpha,beta] = calcEffectiveSpectraBetaMatrixElemMinus(par,ham,cfg,hFSol,alpha,beta)
            else:
                dos[:,alpha,beta] = dos[:,beta,alpha]
    
    return w, dos


def calcEffectiveSpectraBeta(par,ham,cfg,hFSol):
    
    edSol = io.fetchLastEDSolution(cfg)
    eigVals = edSol['eigValsFull']
    eigVecs = edSol['eigVecsFull']
    
    # set up uncorrelated rest Problem:
    print(cfg['tempFileConstrain'])
    if os.path.isfile(cfg['tempFileConstrain']):
        _, coeffsD, coeffsC1 = io.getRecentOnePartBasis(cfg)
    else:
        raise Exception('no opt. one particle basis found!')
    for iO in range(par.NimpOrbs):
        coeffsC1[iO,:] /= np.sqrt(np.sum(coeffsC1[iO,:]**2))
        coeffsD[iO,:] /= np.sqrt(np.sum(coeffsD[iO,:]**2))
    onePartHamilUnCorrSub,onePartHamilUnCorrSubOrb, coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(cfg,par,hFSol,coeffsD,coeffsC1)

    # solve the uncorrelated Problem orbitlly blockwise:
    unCorrEigVals,unCorrEigVecs,_,_,_, energyEffUnCorr = solveUncorrelatedProblem(par,onePartHamilUnCorrSubOrb)

    unCorrDenMat = np.diag(fermiFunc(par,unCorrEigVals))
    
    w = np.linspace(cfg['ed']['minW'],cfg['ed']['maxW'],cfg['ed']['numW'])
    
    if cfg['ed']['calcSpectra'] == False:
       print('not calculating spectral function!',flush=True)
       dos = np.zeros(shape=w.shape)
    else:
        if cfg['aim']['onlyGroundState'] == True:
            raise Exception('to calculate the spectral functions, do thermodynamic calculation')
        print('calculating spectral function... ',flush=True)
        
        # the thermodynamic cut off is of no use for calculating the spectral function
        
      
        # recalcuate the partition function without cut off
        ZZ1Til,ZZ1Block,E0,minIndex,E0Index,thermo = partitionFunctionSpectra(par,ham,eigVals)
        
        # store the matrices in short variable names
        creaImp = ham['oper']['+imp1_up']
        anniImp = ham['oper']['-imp1_up']
        creaBath = ham['oper']['+bath1_1_up']
        anniBath = ham['oper']['-bath1_1_up']
        
        # store the quantum numbers list in an array
        quantumNums = np.zeros(shape=(len(ham['diagblk_qnum']),len(ham['diagblk_qnum'][0])),dtype=int)
    
        for ii in range(len(ham['diagblk_qnum'])):
            quantumNums[ii,0] = ham['diagblk_qnum'][ii][0]
            quantumNums[ii,1] = ham['diagblk_qnum'][ii][1]
            quantumNums[ii,2] = ham['diagblk_qnum'][ii][2]
        
        # store the spectrum in here
        spec = np.zeros(shape=(w.size),dtype=complex)

        #
        # uncorrelated part of the spectrum:
        #

        # transformation from eigenbasis of the uncorrelated basis to
        # first guess, orthogonal basis in which hamitlonian is projected
        # and then to the original basis
        uVAll = np.dot(coeffsOrbComplete[0,0,2*par.NimpOrbs:],unCorrEigVecs)

        # sum over all eigenenergies of the uncorrelated system
        # weigthed with the overlap to the d state
        for ik in range(unCorrEigVals.size):
            spec += (uVAll[ik]*uVAll[ik].conjugate()) / (w + (-unCorrEigVals[ik] - 1j*cfg['ed']['idelta']))

        #
        # correlated part of the spectrum
        #

        # loop over all degenerate ground state states
        for iBlock in range(len(eigVals)):
            print('\n block',quantumNums[iBlock,0],'importance',thermo[iBlock])
            for iVal in range(eigVals[iBlock].size):
                nVec = eigVecs[iBlock][:,iVal]
                En = eigVals[iBlock][iVal]
                tempN = ZZ1Block[iBlock][iVal]
                if tempN < cfg['ed']['betaCutOffExpecValue']:
                    continue
                
                # find the block in which we will land:
                booleanPartNumberCrea = quantumNums[:,0] == 1+ ham['diagblk_qnum'][iBlock][0]
                booleanSzDnNumberCrea = quantumNums[:,1] ==  ham['diagblk_qnum'][iBlock][1]
                booleanSzUpNumberCrea = quantumNums[:,2] == 1+ ham['diagblk_qnum'][iBlock][2]
                # all quantum numbers have to fit
                booleanCrea = (np.vstack((booleanPartNumberCrea,booleanSzUpNumberCrea,booleanSzDnNumberCrea)).all(axis=0))
                
                mdn = []
                mdnSq = []
                for jBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
            
                    mVecs = eigVecs[jBlock]
                    matBlockImp = creaImp[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBath = creaBath[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    
                    mdnImp = np.sum(matBlockImp.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnBath = np.sum(matBlockBath.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdn.append(coeffsD[0,0]*mdnImp + coeffsC1[0,0]*mdnBath)
                
                count = 0
                for jBlock in np.arange(quantumNums.shape[0])[booleanCrea]:
                    Em =eigVals[jBlock]
                    spec += tempN*np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1
                    
                booleanPartNumberAnni = quantumNums[:,0] == -1+ ham['diagblk_qnum'][iBlock][0]
                booleanSzDnNumberAnni = quantumNums[:,1] ==  ham['diagblk_qnum'][iBlock][1]
                booleanSzUpNumberAnni = quantumNums[:,2] == -1+ ham['diagblk_qnum'][iBlock][2]
                # all quantum numbers have to fit
                booleanAnni = (np.vstack((booleanPartNumberAnni,booleanSzUpNumberAnni,booleanSzDnNumberAnni)).all(axis=0))
                
                mdn = []
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    mVecs = eigVecs[jBlock]
                    matBlockImp = anniImp[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBath = anniBath[ham['diagblk_ind'][jBlock]:ham['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    
                    mdnImp = np.sum(matBlockImp.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnBath = np.sum(matBlockBath.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    
                    mdn.append(coeffsD[0,0]*mdnImp + coeffsC1[0,0]*mdnBath)
            
                count = 0
                for jBlock in np.arange(quantumNums.shape[0])[booleanAnni]:
                    Em =eigVals[jBlock]
                    spec += tempN*np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (-En + Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1                               
        dos = 1.0/np.pi*spec.imag/ZZ1Til
    return w, dos
    
def calcEffectiveSpectraBetaClever(par,ham,hamP1,hamM1,cfg,hFSol,nGS):
    
    cfg['ed']['excited'] = 0
    edSol = io.fetchLastEDSolution(cfg)
    eigVals = edSol['eigValsFull']
    eigVecs = edSol['eigVecsFull']
    
    cfg['ed']['excited'] = 1
    edSolP1 = io.fetchLastEDSolution(cfg)
    eigValsP1 = edSolP1['eigValsFull']
    eigVecsP1 = edSolP1['eigVecsFull']
    
    cfg['ed']['excited'] = -1
    edSolM1 = io.fetchLastEDSolution(cfg)
    eigValsM1 = edSolP1['eigValsFull']
    eigVecsM1 = edSolP1['eigVecsFull']
    #print(eigV)
    # set up uncorrelated rest Problem:
    print(cfg['tempFileConstrain'])
    if os.path.isfile(cfg['tempFileConstrain']):
        _, coeffsD, coeffsC1 = io.getRecentOnePartBasis(cfg)
    else:
        raise Exception('no opt. one particle basis found!')
    for iO in range(par.NimpOrbs):
        coeffsC1[iO,:] /= np.sqrt(np.sum(coeffsC1[iO,:]**2))
        coeffsD[iO,:] /= np.sqrt(np.sum(coeffsD[iO,:]**2))
    onePartHamilUnCorrSub,onePartHamilUnCorrSubOrb, coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(cfg,par,hFSol,coeffsD,coeffsC1)

    # solve the uncorrelated Problem orbitlly blockwise:
    unCorrEigVals,unCorrEigVecs,_,_,_, energyEffUnCorr = solveUncorrelatedProblem(par,onePartHamilUnCorrSubOrb)

    unCorrDenMat = np.diag(fermiFunc(par,unCorrEigVals))
    
    w = np.linspace(cfg['ed']['minW'],cfg['ed']['maxW'],cfg['ed']['numW'])
    
    if cfg['ed']['calcSpectra'] == False:
       print('not calculating spectral function!',flush=True)
       dos = np.zeros(shape=w.shape)
    else:
        if cfg['aim']['onlyGroundState'] == True:
            raise Exception('to calculate the spectral functions, do thermodynamic calculation')
        print('calculating spectral function... ',flush=True)
        
        # the thermodynamic cut off is of no use for calculating the spectral function
        
      
        # recalcuate the partition function without cut off
        ZZ1Til,ZZ1Block,E0,minIndex,E0Index,thermo = partitionFunctionSpectra(par,ham,eigVals)
        
        # store the matrices in short variable names
        creaImp = ham['oper']['+imp1_up']
        anniImp = ham['oper']['-imp1_up']
        creaBath = ham['oper']['+bath1_1_up']
        anniBath = ham['oper']['-bath1_1_up']
        
        # store the spectrum in here
        spec = np.zeros(shape=(w.size),dtype=complex)

        #
        # uncorrelated part of the spectrum:
        #

        # transformation from eigenbasis of the uncorrelated basis to
        # first guess, orthogonal basis in which hamitlonian is projected
        # and then to the original basis
        uVAll = np.dot(coeffsOrbComplete[0,0,2*par.NimpOrbs:],unCorrEigVecs)

        # sum over all eigenenergies of the uncorrelated system
        # weigthed with the overlap to the d state
        for ik in range(unCorrEigVals.size):
            spec += (uVAll[ik]*uVAll[ik].conjugate()) / (w + (-unCorrEigVals[ik] - 1j*cfg['ed']['idelta']))

        #
        # correlated part of the spectrum
        #

        # loop over all degenerate ground state states
        for iBlock in range(len(eigVals)):
            if nGS != ham['diagblk_qnum'][iBlock][0]:
                continue
            print('\n block',ham['diagblk_qnum'][iBlock][0],'importance',thermo[iBlock])
            for iVal in range(eigVals[iBlock].size):
                nVec = eigVecs[iBlock][:,iVal]
                En = eigVals[iBlock][iVal]
                tempN = ZZ1Block[iBlock][iVal]
                if tempN < cfg['ed']['betaCutOffExpecValue']:
                    continue
                
                # find the block in which we will land:
                #booleanPartNumberCrea = quantumNums[:,0] == 1+ ham['diagblk_qnum'][iBlock][0]
                #booleanSzDnNumberCrea = quantumNums[:,1] ==  ham['diagblk_qnum'][iBlock][1]
                #booleanSzUpNumberCrea = quantumNums[:,2] == 1+ ham['diagblk_qnum'][iBlock][2]
                # all quantum numbers have to fit
                #booleanCrea = (np.vstack((booleanPartNumberCrea,booleanSzUpNumberCrea,booleanSzDnNumberCrea)).all(axis=0))
                
                mdn = []
                mdnSq = []
                for jBlock in np.arange(len(hamP1['diagblk_qnum'])):
            
                    mVecs = eigVecsP1[jBlock]
                    matBlockImp = creaImp[hamP1['diagblk_ind'][jBlock]:hamP1['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBath = creaBath[hamP1['diagblk_ind'][jBlock]:hamP1['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    
                    mdnImp = np.sum(matBlockImp.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnBath = np.sum(matBlockBath.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdn.append(coeffsD[0,0]*mdnImp + coeffsC1[0,0]*mdnBath)
                
                count = 0
                for jBlock in np.arange(len(hamP1['diagblk_qnum'])):
                    Em =eigValsP1[jBlock]
                    spec += tempN*np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1
                    
                
                mdn = []
                for jBlock in np.arange(len(hamM1['diagblk_qnum'])):
                    mVecs = eigVecsM1[jBlock]
                    matBlockImp = anniImp[hamM1['diagblk_ind'][jBlock]:hamM1['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    matBlockBath = anniBath[hamM1['diagblk_ind'][jBlock]:hamM1['diagblk_ind'][jBlock+1],ham['diagblk_ind'][iBlock]:ham['diagblk_ind'][iBlock+1]]
                    
                    mdnImp = np.sum(matBlockImp.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    mdnBath = np.sum(matBlockBath.dot(nVec)[:,np.newaxis]*mVecs,axis=0)
                    
                    mdn.append(coeffsD[0,0]*mdnImp + coeffsC1[0,0]*mdnBath)
            
                count = 0
                for jBlock in np.arange(len(hamM1['diagblk_qnum'])):
                    Em =eigValsM1[jBlock]
                    spec += tempN*np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (-En + Em - 1j*cfg['ed']['idelta'])),axis=1)       
                    count += 1                               
        dos = 1.0/np.pi*spec.imag/ZZ1Til
    return w, dos
    
def calcSpectra(par,ham,blockCar,hFSol):
    #%% 

    edSol = io.fetchLastEDSolution(cfg)
        
    # set up uncorrelated rest Problem:
    if os.path.isfile(par.tempFileConstrain+'.pkl'):
        _, coeffsD, coeffsC1 = io.getRecentOnePartBasis(par)
    else:
        raise Exception('no opt. one particle basis found!')
    for iO in range(par.NimpOrbs):
        coeffsC1[iO,:] /= np.sqrt(np.sum(coeffsC1[iO,:]**2))
        coeffsD[iO,:] /= np.sqrt(np.sum(coeffsD[iO,:]**2))
    onePartHamilUnCorrSub , onePartHamilUnCorrSubOrb ,coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(par,hFSol,coeffsD,coeffsC1)
    
    # solve the uncorrelated Problem orbitlly blockwise:
    unCorrEigVals,_,_,_,_, energyEffUnCorr = solveUncorrelatedProblem(par,onePartHamilUnCorrSubOrb)
    unCorrDenMat = np.diag(fermiFunc(par,unCorrEigVals))


    # do some preliminary stuff: quantum numbers; get nature of GS

    quantumNums = np.zeros(shape=(len(ham['diagblk_qnum']),len(ham['diagblk_qnum'][0])),dtype=int)
    
    for ii in range(len(ham['diagblk_qnum'])):
        quantumNums[ii,0] = ham['diagblk_qnum'][ii][0]
        quantumNums[ii,1] = ham['diagblk_qnum'][ii][1]
        quantumNums[ii,2] = ham['diagblk_qnum'][ii][2]

        
    # lowest energy
    minList = [np.min(blockEner) for blockEner in edSol['eigVals']]
    E0Prime = min(minList)
    # now find all blocks in which groundstate energy exists
    minBlockIndex = [jj for jj, vv in enumerate(minList) if np.abs(vv - E0Prime)<1e-14]
    E0 = []
    E0Index = []
    NG = ham['diagblk_qnum'][minBlockIndex[0]][0]
    # now get all groundstate energies (and their indices) in the blocks
    for ii in minBlockIndex:
        dummy = []
        for jj in range(edSol['eigVals'][ii].size):
            if np.abs(edSol['eigVals'][ii][jj]-E0Prime)<1e-14:
                E0.append(edSol['eigVals'][ii][jj])
                dummy.extend([jj])
                #E0Index.append([ii, jj])
        E0Index.append(dummy)
    # calculate <m|d|n>
    Dim = len(ham['diagblk_dim'])
    indizes = np.zeros(Dim+1,dtype=int)
    indizes[1:] = np.cumsum(ham['diagblk_dim'])
    matD = ham['oper']['+imp1_up']
    matC = ham['oper']['+bath1_1_up']
    w = np.linspace(-10.0,10.0,2000)
    idelta = 0.1
    spec = np.zeros(shape=(w.size,4),dtype=complex)
    mcknT = []
    
    for ik in range(unCorrEigVals.size):
        mcknT = (coeffsComplete[0,2*par.NimpOrbs+ik] * (1.0 - unCorrDenMat[ik,ik]))
        

        spec[:,3] += (mcknT*mcknT.conjugate()) / (w + (-unCorrEigVals[ik] - 1j*idelta))
    
    for iDeg in range(len(E0Index)):
        for jDeg in range(len(E0Index[iDeg])):

            nVec = edSol['eigVecs'][minBlockIndex[iDeg]][:,E0Index[iDeg][jDeg]]
            #print('quantumnumbers of GS block:',ham['diagblk_qnum'][minBlockIndex[iDeg]])
            booleanPartNumber = quantumNums[:,0] == 1+ ham['diagblk_qnum'][minBlockIndex[iDeg]][0]
            booleanSzDnNumber = quantumNums[:,1] ==  ham['diagblk_qnum'][minBlockIndex[iDeg]][1]
            booleanSzUpNumber = quantumNums[:,2] == 1+ ham['diagblk_qnum'][minBlockIndex[iDeg]][2]
            boolean = (np.vstack((booleanPartNumber,booleanSzUpNumber,booleanSzDnNumber)).all(axis=0))
            #boolean= booleanPartNumber
            mdnT = []
            mcnT = []
            mdn = []
            
            for iBlock in np.arange(quantumNums.shape[0])[boolean]:
                mVecs = edSol['eigVecs'][iBlock]
                matBlockD = matD[indizes[iBlock]:indizes[iBlock+1],indizes[minBlockIndex[iDeg]]:indizes[minBlockIndex[iDeg]+1]]
                matBlockC = matC[indizes[iBlock]:indizes[iBlock+1],indizes[minBlockIndex[iDeg]]:indizes[minBlockIndex[iDeg]+1]]

                mdnT.append(np.sum(matBlockD.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
                mcnT.append(np.sum(matBlockC.dot(nVec)[:,np.newaxis]*mVecs,axis=0))
                
                mdnDummy = coeffsD[0,0]*mdnT[-1] + coeffsC1[0,0]*mcnT[-1]
                
                
                mdn.append(mdnDummy)
                #(np.sum(matBlock.dot(mVecs)*eigVecs[count],axis=0))
        
            #print(mVecs.shape)
    
            En = E0[iDeg]
            count = 0
            for iBlock in np.arange(quantumNums.shape[0])[boolean]:
                
                Em = edSol['eigVals'][iBlock]

                spec[:,0] += np.sum((mdnT[count]*mdnT[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*idelta)),axis=1)
                spec[:,1] += np.sum((mcnT[count]*mcnT[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*idelta)),axis=1)
                spec[:,2] += np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*idelta)),axis=1)
                spec[:,3] += np.sum((mdn[count]*mdn[count].conjugate())[np.newaxis,:] / (w[:,np.newaxis] + (En - Em - 1j*idelta)),axis=1)
                
                
                count += 1
    

    #energyOne,energyTwo,energy, completeDenMatLocBasis,twoPartLocUpDn, twoPartLocUpUp = stitchEnergy(par,edSol['corrDenMat'],edSol['twoPart'],unCorrDenMatOrb,coeffsComplete,coeffsOrbComplete)
    #energyEff = energyEffUnCorr + edSol['energyEffCorr']

    return w, spec
    
    
    
 
def stitchEnergy(cfg,par,corrDenMat,twoPart,unCorrDenMatOrb,coeffsComplete,coeffsOrbComplete):
    
    #%%
    ################    
    # calculate one particle part of <H>*
    ################
    
    # density matrix of the uncorrelated part in original basis
    #unCorrDenMatLocBasis = np.dot(coeffsComplete[:,2*par.NimpOrbs:],np.dot(unCorrDenMat,coeffsComplete[:,2*par.NimpOrbs:].transpose()))
    
    unCorrDenMatLocBasisOrb = np.zeros(shape=(par.NimpOrbs,par.Nbath+1, par.Nbath+1))
    
    if par.numDeg == 0:
        for iO in range(par.NimpOrbs):
            unCorrDenMatLocBasisOrb[iO,:,:] = np.dot(coeffsOrbComplete[iO,:,2*par.NimpOrbs:],np.dot(unCorrDenMatOrb[iO,:,:],coeffsOrbComplete[iO,:,2*par.NimpOrbs:].transpose()))
    else:
        for iD in range(par.numDeg):
            #print coeffsOrbComplete[0,:,2:].shape
            #print unCorrDenMatOrb.shape
            dummy = np.dot(coeffsOrbComplete[par.degeneracy[iD][0],:,2:],np.dot(unCorrDenMatOrb[par.degeneracy[iD][0],:,:],coeffsOrbComplete[par.degeneracy[iD][0],:,2:].transpose()))
            for iOD in range(len(par.degeneracy[iD])):
                unCorrDenMatLocBasisOrb[par.degeneracy[iD][iOD],:,:]  = dummy
    unCorrDenMatLocBasis = np.zeros(shape=(par.NimpOrbs*(par.Nbath+1),par.NimpOrbs*(par.Nbath+1)))
    for iO in range(par.NimpOrbs):
        unCorrDenMatLocBasis[iO,iO] = unCorrDenMatLocBasisOrb[iO,0,0]
        unCorrDenMatLocBasis[iO,par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath] = unCorrDenMatLocBasisOrb[iO,0,1:]
        unCorrDenMatLocBasis[par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath,iO] = unCorrDenMatLocBasisOrb[iO,1:,0]
        unCorrDenMatLocBasis[par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath,par.NimpOrbs+iO*par.Nbath:par.NimpOrbs+(iO+1)*par.Nbath] = unCorrDenMatLocBasisOrb[iO,1:,1:]
    #np.savetxt('densmatOrrBasis_uncorr.dat',unCorrDenMatLocBasis)
    # density matrix of the correlated part in original basis
    corrDenMatLocBasis =  np.dot(coeffsComplete[:,:2*par.NimpOrbs],np.dot(corrDenMat,coeffsComplete[:,:2*par.NimpOrbs].transpose()))

    # complete density matrix of effectve system in original basis
    completeDenMatLocBasis = unCorrDenMatLocBasis + corrDenMatLocBasis

    # original one-part. Hamiltonian in original basis
    origOnePartHamil = harFock.onePartMatrix(par.epsBath,par.vBath,np.diag(par.epsImp),par.epsBath.shape[1])

    # expectation value of original one particle hamiltonian evaluated with
    # effective system: <H_1>*
    onePartEnergy = 2.0*np.sum(completeDenMatLocBasis * origOnePartHamil)

    ################    
    # calculate two particle part of <H>*
    ################

    if cfg['algo']['optOnlyBath']:
    #if False:
        # in this case twoPart allready contains the only local parts
        # so no transformation is neaded
        twoPartLocUpDn = twoPart['updn']
        twoPartLocDnUp = twoPart['dnup']
        twoPartLocUpUp = twoPart['upup']
        #twoPartLoc = twoPart['updn'] + twoPart['dnup'] + 2*twoPart['upup']
        
    else:
        # first calculate twopart expectation value from the correlated subspace in the original basis
        twoPartCorrLocUpDn = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
        twoPartCorrLocDnUp = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
        twoPartCorrLocUpUp = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
        #twoPartCorrLoc = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
        
        # loop over non-zero elements of par.uMatrixNonLocal    
        #print '2P: calculating',nonzeros[0].size, 'of',par.uMatrixNonLocal.size
        
        for ii in range(par.NimpOrbs):
            for jj in range(par.NimpOrbs):
                for kk in range(par.NimpOrbs):
                    for ll in range(par.NimpOrbs):
                        if par.uMatrix[ii,jj,ll,kk] != 0:                            
                            for ii1 in range(2):
                                for jj1 in range(2):
                                    for kk1 in range(2):
                                        for ll1 in range(2):
                                            factor =  coeffsOrbComplete[ii,0,ii1] * coeffsOrbComplete[jj,0,jj1] * coeffsOrbComplete[kk,0,kk1]* coeffsOrbComplete[ll,0,ll1]
                                            twoPartCorrLocUpDn[ii,jj,kk,ll] += twoPart['updn'][ii+ii1*par.NimpOrbs,jj+jj1*par.NimpOrbs,kk+kk1*par.NimpOrbs,ll+ll1*par.NimpOrbs] * factor
                                            twoPartCorrLocDnUp[ii,jj,kk,ll] += twoPart['dnup'][ii+ii1*par.NimpOrbs,jj+jj1*par.NimpOrbs,kk+kk1*par.NimpOrbs,ll+ll1*par.NimpOrbs] * factor
                                            twoPartCorrLocUpUp[ii,jj,kk,ll] += twoPart['upup'][ii+ii1*par.NimpOrbs,jj+jj1*par.NimpOrbs,kk+kk1*par.NimpOrbs,ll+ll1*par.NimpOrbs] * factor


        # now stitch together parts from correlated and uncorrelated part:                
        twoPartLocUpDn = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
        twoPartLocDnUp = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
        twoPartLocUpUp = np.zeros(shape=(par.NimpOrbs,par.NimpOrbs,par.NimpOrbs,par.NimpOrbs))
        
        # the fock-terms do not contribute in the updn part (because of spin-diagonality)
        for ii in range(par.NimpOrbs):
            for jj in range(par.NimpOrbs):
                for kk in range(par.NimpOrbs):
                    for ll in range(par.NimpOrbs):
                        if par.uMatrix[ii,jj,ll,kk] != 0:
                            twoPartLocUpUp[ii,jj,kk,ll] = ( twoPartCorrLocUpUp[ii,jj,kk,ll]
                                          -corrDenMatLocBasis[ii,kk]*unCorrDenMatLocBasis[jj,ll]
                                          +corrDenMatLocBasis[ii,ll]*unCorrDenMatLocBasis[jj,kk]
                                          +unCorrDenMatLocBasis[ii,ll]*corrDenMatLocBasis[jj,kk]
                                          -unCorrDenMatLocBasis[ii,kk]*corrDenMatLocBasis[jj,ll]
                                          +unCorrDenMatLocBasis[ii,ll]*unCorrDenMatLocBasis[jj,kk]
                                          -unCorrDenMatLocBasis[ii,kk]*unCorrDenMatLocBasis[jj,ll])
                            # Fock-Terms vanish for UpDn, because no spin-flip is considered
                            twoPartLocUpDn[ii,jj,kk,ll] = ( twoPartCorrLocUpDn[ii,jj,kk,ll]
                                          +corrDenMatLocBasis[ii,ll]*unCorrDenMatLocBasis[jj,kk]
                                          +unCorrDenMatLocBasis[ii,ll]*corrDenMatLocBasis[jj,kk]
                                          +unCorrDenMatLocBasis[ii,ll]*unCorrDenMatLocBasis[jj,kk])
                            twoPartLocDnUp[ii,jj,kk,ll] = ( twoPartCorrLocDnUp[ii,jj,kk,ll]
                                          +corrDenMatLocBasis[ii,ll]*unCorrDenMatLocBasis[jj,kk]
                                          +unCorrDenMatLocBasis[ii,ll]*corrDenMatLocBasis[jj,kk]
                                          +unCorrDenMatLocBasis[ii,ll]*unCorrDenMatLocBasis[jj,kk])
            
                     
                         
    # factor 1/2 cancels in twoPartLocUpUp, because here only UpUp is included (and not DnDn)
    # twopart energy
    twoPartEnergy = 0.0
    for ii in range(par.NimpOrbs):
        for jj in range(par.NimpOrbs):
            for kk in range(par.NimpOrbs):
                for ll in range(par.NimpOrbs):
                    if par.uMatrix[ii,jj,ll,kk] != 0:
                        twoPartEnergy += par.uMatrix[ii,jj,ll,kk] * (
                                                     0.5*twoPartLocUpDn[ii,jj,kk,ll]
                                                    +0.5*twoPartLocDnUp[ii,jj,kk,ll]
                                                    +twoPartLocUpUp[ii,jj,kk,ll]
                                                             )
    energy = onePartEnergy + twoPartEnergy
        
    return onePartEnergy,twoPartEnergy,energy, completeDenMatLocBasis, twoPartLocUpDn, twoPartLocUpUp

def funcEDResTest(par,parEff,spPars,car,ham):
    start = time.time()
    print('setting H... ',end=' ',flush=True)
    H = set_ham_pure(parEff,ham)
    print('took {:1.2f} sec'.format(time.time() - start),flush=True)
    start = time.time()
    print('solving H... ',end=' ',flush=True)
    eigVals,eigVecs = diagonalize_blocks(H,ham['diagblk_qnum'],ham['diagblk_dim'],ham['diagblk_ind'],'sparse',cfg['ed'])
    print('took {:1.2f} sec'.format(time.time() - start),flush=True)

    cfg['ed']['restric'] = True
    for ii in range(20):
        start = time.time()
        #par.betaCutOffExpecValue = 10**(-(ii+1))
        #par.PMBlock = ii+1
        #parEff.PMBlock = ii+1
        minList = [np.min(blockEner) for blockEner in eigVals]
        # lowest energy                                                            
        E0Prime = min(minList)
        # now find all blocks in which groundstate energy exists               
        minBlockIndex = [jj for jj, vv in enumerate(minList) if np.abs(vv - E0Prime)<1e-14]
        E0 = []
        E0Index = []
        # particle number in ground state (T=0):
        NG = ham['diagblk_qnum'][minBlockIndex[0]][0]

        # now get all groundstate energies (and their indices) in the blocks
        for ii in minBlockIndex:
            dummy = []
            for jj in range(eigVals[ii].size):
                if np.abs(eigVals[ii][jj]-E0Prime)<1e-14:
                    E0.append(eigVals[ii][jj])
                    dummy.extend([jj])
            E0Index.append(dummy)

        ZZ1BlockTil = []
        ZZ1Til = 0
        ZZ1TilRes = 0
        # loop over subspaces, do not sum, because we need these factors also to
        # evaluate the expectaion values. we can reuse them there                        
        for iBlock in range(len(eigVals)):
            ZZ1BlockTil.append(np.exp(-parEff.beta * ( eigVals[iBlock] - E0[0] - parEff.mu*ham['diagblk_qnum'][iBlock][0] )))
            sumBlock = np.sum(ZZ1BlockTil[iBlock])
            ZZ1Til += sumBlock
            if abs(ham['diagblk_qnum'][iBlock][0] - NG) < (par.PMBlock+1):
                ZZ1TilRes += sumBlock



        Phi = (-1/parEff.beta*np.log(ZZ1Til)) + E0[0]
        for jjj in range(100):
            Res = calcExpecValueRes(eigVecs,ham['diagblk_dim'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1TilRes,ham['oper']['+imp'+str(1)+'_up'] * ham['oper']['-imp'+str(1)+'_up'],cfg,parEff)
        norm = calcExpecValue(eigVecs,ham['diagblk_dim'],ZZ1BlockTil,ZZ1Til, ham['oper']['+imp'+str(1)+'_up']*ham['oper']['-imp'+str(1)+'_up'],parEff)

        print( 'Res', Res)
        print( 'norm', norm)
        print( 'diff', Res - norm)
        print( time.time() - start)







def funcEDPhi(cfg,par,parEff,car,ham):
    #%% 
    start = time.time()
    if par.NimpOrbs > 1:
        print('setting H... ',end='',flush=True)
    #print(parEff.epsImp)
    #print(parEff.vBath)
    #print(parEff.epsBath)
    
    H = set_ham_pure(parEff,ham)
    if par.NimpOrbs > 1:
        print('took {:1.2f} sec'.format(time.time() - start),flush=True)
    start = time.time()
    if par.NimpOrbs > 1:
        print('solving H... ',end='',flush=True)
    eigVals,eigVecs = diagonalize_blocks(H,ham['diagblk_qnum'],ham['diagblk_dim'],ham['diagblk_ind'],'sparse',cfg['ed'])
    eigValsFull = copy.copy(eigVals)
    eigVecsFull = copy.copy(eigVecs)
#nPart = np.arange(0,par.NfStates)
    #eigVals,eigVecs = diagonalize_blocks_part(H,ham['diagblk_qnum'],ham['diagblk_dim'],'sparse',cfg['ed'],nPart)
    print('took {:1.2f} sec'.format(time.time() - start),flush=True)
    start = time.time()
    ZZ1Til,ZZ1BlockTil,Phi,NG,E0,eigVecs,thermo,_,_ = partitionFunction(par,ham,eigVals,eigVecs,cfg)

    start = time.time()
    if par.NimpOrbs > 1:
        print('calculating expectation values... ',end='',flush=True)
    
    # calculation of <i|c^+c|j>     
    den1 = np.zeros(shape=(2,parEff.NimpOrbs,parEff.Nbath))
    for iO in range(par.NimpOrbs):
        for iB in range(parEff.Nbath):
            # up
            ham['bathUp'][iO][iB][0:1,0:1]
            den1[0,iO,iB] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['bathUp'][iO][iB],cfg,parEff,thermo)
            den1[1,iO,iB] = den1[0,iO,iB]

            
    # calculation of <i|c^+d|j>
    cd = np.zeros(shape=(4,parEff.NimpOrbs,parEff.Nbath)) 

    for iO in range(parEff.NimpOrbs):
        for iB in range(parEff.Nbath):
            # UpUp
            cd[0,iO,iB] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['hybUp'][iO][iB],cfg,parEff,thermo)
            # UpDn
            cd[1,iO,iB] = 0.0
            # DnUp
            cd[2,iO,iB] = 0.0
            # DnDn
            cd[3,iO,iB] = cd[0,iO,iB]
    
    # calculation of <i|d^+d|j> 
    denImp = np.zeros(shape=(2,parEff.NimpOrbs))
    for ii in range(parEff.NimpOrbs):
        # up
        denImp[0,ii] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+imp'+str(ii+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up'],cfg,parEff,thermo)
        denImp[1,ii] = denImp[0,ii]
    print('electrons on the eff. imp.:',np.sum(denImp))
    print('total electrons int the system:',np.sum(denImp)+np.sum(den1))
    twoPartN = dict()   
    if cfg['algo']['optOnlyBath']:
        dim = parEff.NimpOrbs
    else:
        dim = 2*parEff.NimpOrbs
    #print(denImp[0,:]*2.0)
    #print(den1[0,:,:]*2.0)
    #print(cd[0,:,:])
    if par.NimpOrbs > 1:
        print('finished one part.',end=' ',flush=True)
    twoPartN['upup'] = np.zeros(shape=(dim,dim,dim,dim))
    twoPartN['updn'] = np.zeros(shape=(dim,dim,dim,dim))
    twoPartN['dnup'] = np.zeros(shape=(dim,dim,dim,dim))
    twoPartN['dndn'] = np.zeros(shape=(dim,dim,dim,dim)) 
    if cfg['algo']['optOnlyBath']:
        #twoPartN['upup'][...] = float('nan')
        #twoPartN['updn'][...] = float('nan')
        #twoPartN['dnup'][...] = float('nan')
        #twoPartN['dndn'][...] = float('nan')
        for ii in range(dim):
            for jj in range(dim):
                for kk in range(dim):
                    for ll in range(dim):
                        # upup
                        #twoPartN['upup'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+'+str(ii)+'_up']*ham['oper']['+'+str(jj)+'_up']*ham['oper']['-'+str(kk)+'_up']*ham['oper']['-'+str(ll)+'_up'],cfg,parEff)
                        #twoPartN['updn'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+'+str(ii)+'_up']*ham['oper']['+'+str(jj)+'_dn']*ham['oper']['-'+str(kk)+'_dn']*ham['oper']['-'+str(ll)+'_up'],cfg,parEff)
                        #twoPartN['dnup'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+'+str(ii)+'_dn']*ham['oper']['+'+str(jj)+'_up']*ham['oper']['-'+str(kk)+'_up']*ham['oper']['-'+str(ll)+'_dn'],cfg,parEff)
                        
                        if True:
                            
                            if (ii == jj) or (kk == ll):
                                twoPartN['upup'][ii,jj,kk,ll] = 0.0
                            elif (ii > jj):
                                twoPartN['upup'][ii,jj,kk,ll] = -twoPartN['upup'][jj,ii,kk,ll]
                            elif (ll > kk):
                                # warning: 
                                #twoPartN['upup'][ii,jj,kk,ll] = -twoPartN['upup'][ii,jj,ll,kk]
                                # the ii,jj,ll,kk are not calculated yet.
                                # thats why we only pass here and put the later calculated
                                # values in an additional loop afterwards
                                pass
                               
                            else:
                                if ((ii == kk) and (jj == ll)) or (parEff.uMatrix[ii,jj,ll,kk] != 0):
                                    cPcP1 = ham['oper']['++'+str(ii)+'_'+str(jj)+'_up'+'_up']
                                    cMcM1 = ham['oper']['++'+str(ll)+'_'+str(kk)+'_up'+'_up'].transpose().conjugate()
                                    twoPartN['upup'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,cPcP1*cMcM1,cfg, parEff,thermo)                                               
                            
                            # up down
                            if ((ii == ll) and (jj == kk)) or ((ii == kk) and (jj == ll))  or (parEff.uMatrix[ii,jj,ll,kk] != 0):
                                cPcP2 = ham['oper']['++'+str(ii)+'_'+str(jj)+'_up'+'_dn']
                                cMcM2 = ham['oper']['++'+str(ll)+'_'+str(kk)+'_up'+'_dn'].transpose().conjugate()
                                twoPartN['updn'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,cPcP2*cMcM2,cfg,parEff,thermo)
        
    else:

        for ii in range(dim):
            for jj in range(dim):
                for kk in range(dim):
                    for ll in range(dim):
                        #twoPartN['upup'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+'+str(ii)+'_up']*ham['oper']['+'+str(jj)+'_up']*ham['oper']['-'+str(kk)+'_up']*ham['oper']['-'+str(ll)+'_up'],cfg,parEff)
                        #twoPartN['updn'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+'+str(ii)+'_up']*ham['oper']['+'+str(jj)+'_dn']*ham['oper']['-'+str(kk)+'_dn']*ham['oper']['-'+str(ll)+'_up'],cfg,parEff)
                        #twoPartN['dnup'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,ham['oper']['+'+str(ii)+'_dn']*ham['oper']['+'+str(jj)+'_up']*ham['oper']['-'+str(kk)+'_up']*ham['oper']['-'+str(ll)+'_dn'],cfg,parEff)
                        
                        if True:
                            # upup
                            if (ii == jj) or (kk == ll):
                                twoPartN['upup'][ii,jj,kk,ll] = 0.0
                            elif (ii > jj):
                                twoPartN['upup'][ii,jj,kk,ll] = -twoPartN['upup'][jj,ii,kk,ll]
                            elif (ll > kk):
                                # warning: 
                                #twoPartN['upup'][ii,jj,kk,ll] = -twoPartN['upup'][ii,jj,ll,kk]
                                # the ii,jj,ll,kk are not calculated yet.
                                # thats why we only pass here and put the later calculated
                                # values in an additional loop afterwards
                                pass
                               
                            else:
                                cPcP1 = ham['oper']['++'+str(ii)+'_'+str(jj)+'_up'+'_up']
                                cMcM1 = ham['oper']['++'+str(ll)+'_'+str(kk)+'_up'+'_up'].transpose().conjugate()
                                twoPartN['upup'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,cPcP1*cMcM1,cfg, parEff,thermo)                                               
                            
                            # up down
                            cPcP2 = ham['oper']['++'+str(ii)+'_'+str(jj)+'_up'+'_dn']
                            cMcM2 = ham['oper']['++'+str(ll)+'_'+str(kk)+'_up'+'_dn'].transpose().conjugate()
                            twoPartN['updn'][ii,jj,kk,ll] = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,cPcP2*cMcM2,cfg,parEff,thermo)

    
    
    if True:
        for ii in range(dim):
            for jj in range(dim):
                for kk in range(dim):
                    for ll in range(dim):
                        if (ll > kk):
                            twoPartN['upup'][ii,jj,kk,ll] = -twoPartN['upup'][ii,jj,ll,kk]
                        twoPartN['dnup'][ii,jj,kk,ll] = twoPartN['updn'][jj,ii,ll,kk].copy()
    

    #for ii in range(dim):
    #        for jj in range(dim):
    #            for kk in range(dim):
    #                for ll in range(dim):
    #                    value = twoPartN['upup'][ii,jj,kk,ll]+twoPartN['dnup'][ii,jj,kk,ll]+twoPartN['updn'][ii,jj,kk,ll]+twoPartN['dndn'][ii,jj,kk,ll]
    #                    if np.abs(value) > 1e-15:
    #                        print(ii,jj,kk,ll,value)

    
    twoPartN['dndn'] = copy.copy(twoPartN['upup'])
    #print('twoPartN',twoPartN)
    if par.NimpOrbs > 1:
        print('finished two part.',end=' ',flush=True)
    energy = calcExpecValue(eigVecs,ham['diagblk_ind'],ZZ1BlockTil,ZZ1Til,H,parEff)
    if par.NimpOrbs > 1:
        print('took {:1.2f} sec'.format(time.time() - start),flush=True)

    
    # charge fluctuation
    # calculate <N^2>
    # inintialize matrix:
    sqrtN2Mat = 0.0*ham['oper']['+imp'+str(1)+'_up']
    
    for ii in range(parEff.NimpOrbs):
        # up
        sqrtN2Mat = sqrtN2Mat + ham['oper']['+imp'+str(ii+1)+'_up'] * ham['oper']['-imp'+str(ii+1)+'_up']
        sqrtN2Mat = sqrtN2Mat + ham['oper']['+imp'+str(ii+1)+'_dn'] * ham['oper']['-imp'+str(ii+1)+'_dn']
    n2Mat = sqrtN2Mat*sqrtN2Mat
    n2 = calcExpecValueRes(eigVecs,ham['diagblk_ind'],ham['diagblk_qnum'],NG,ZZ1BlockTil,ZZ1Til,n2Mat,cfg,parEff,thermo)    
        
    
    corrDenMat = np.zeros(shape=(par.NimpOrbs*2,par.NimpOrbs*2))
    for iO in range(par.NimpOrbs):
        corrDenMat[iO,iO] = denImp[0,iO]
        corrDenMat[iO,iO+par.NimpOrbs] = cd[0,iO,0]
        corrDenMat[iO+par.NimpOrbs,iO] = cd[0,iO,0]
        corrDenMat[iO+par.NimpOrbs,iO+par.NimpOrbs] = den1[0,iO,0]
    #print('corrDenMat',corrDenMat)
    # calculate <S^2> 
    # this is not needed, as it can be calculated from the one and two particle
    # density matrices. It is kept for benchmark here
    #sZ = sp.csr_matrix((2**(parEff.NfStates), 2**(parEff.NfStates)), dtype=float)
    #sP = sp.csr_matrix((2**(parEff.NfStates), 2**(parEff.NfStates)), dtype=float)
    #sM = sp.csr_matrix((2**(parEff.NfStates), 2**(parEff.NfStates)), dtype=float)
    #for iO in range(par.NimpOrbs):
    #    sZ = sZ + 0.5*( ham['oper']['+imp'+str(iO+1)+'_up'] * ham['oper']['-imp'+str(iO+1)+'_up']
    #               -ham['oper']['+imp'+str(iO+1)+'_dn'] * ham['oper']['-imp'+str(iO+1)+'_dn'])
    #    sP = sP + ham['oper']['+imp'+str(iO+1)+'_up'] * ham['oper']['-imp'+str(iO+1)+'_dn']
    #    sM = sM + ham['oper']['+imp'+str(iO+1)+'_dn'] * ham['oper']['-imp'+str(iO+1)+'_up']
    #SSq = 0.5*(sP*sM + sM*sP) + sZ * sZ
    #print('sSquare per expec', calcExpecValue(eigVecs,ham['diagblk_dim'],ZZ1BlockTil,ZZ1Til,SSq,parEff))
    
    edSol = dict()
    edSol['eigVals'] = eigVals
    edSol['eigVecs'] = eigVecs
    edSol['eigValsFull'] = eigValsFull
    edSol['eigVecsFull'] = eigVecsFull
    edSol['corrDenMat'] = corrDenMat
    edSol['n2'] = n2
    edSol['PhiCorr'] = Phi
    edSol['E0'] = E0
    edSol['twoPart'] = twoPartN
    edSol['energyEffCorr'] = energy

    #io.saveEDSolution(cfg,edSol)
    return edSol




def calcEnergyHFFixEDOrb(pointX,*args):
    #%% 

    par = args[0]
    hFSol = args[1]
    edSol = args[2]
    orb = args[3]
    restPoint= args[4]
    cfg = args[5]
    #start = time.time()
    if par.NimpOrbs > 5:
        print('deriv start',end=" ")
    coeffsD, coeffsC1 = pointToVecOrbStitch(cfg,par,pointX,orb,restPoint)
    #print('pointX')
    #print(pointX)
    
    #print np.dot(coeffsC1[0,:],coeffsC1[0,:]), np.dot(coeffsD[0,:],coeffsC1[0,:]), np.dot(coeffsD[0,:],coeffsD[0,:]),
    for iO in range(par.NimpOrbs):
        coeffsC1[iO,:] /= np.sqrt(np.sum(coeffsC1[iO,:]**2))
        coeffsD[iO,:] /= np.sqrt(np.sum(coeffsD[iO,:]**2))
    #print('constrains')
    #print(np.dot(coeffsC1[0,:],coeffsC1[0,:].transpose()))
    #print(np.dot(coeffsD[0,:],coeffsD[0,:].transpose()))
    #print(np.dot(coeffsC1[0,:],coeffsD[0,:].transpose()))
    #print('startstuff',(time.time() - start)*1000.0)
    # set up uncorrelated rest Problem:
    #start = time.time()
    onePartHamilUnCorrSub,onePartHamilUnCorrSubOrb, coeffsOrbComplete, coeffsComplete = harFock.unCorrSubHamiltonian(cfg,par,hFSol,coeffsD,coeffsC1)
    #np.savetxt('matAll.dat',onePartHamilUnCorrSub)
    #for iO in range(3):
    #    np.savetxt('mat'+str(iO)+'.dat',onePartHamilUnCorrSubOrb[iO,:,:])
    #print('setup',(time.time() - start)*1000.0)

    # solve the uncorrelated Problem blockwise:
    #start = time.time()
    _,_,_,unCorrDenMatOrb, PhiUnCorr, energyEffUnCorr  = solveUncorrelatedProblem(par,onePartHamilUnCorrSubOrb)
    #print('solving',(time.time() - start)*1000.0)
    #start = time.time()
    _,_,energy, completeDenMatLocBasis,twoPartLocUpDn, twoPartLocUpUp = stitchEnergy(cfg, par, edSol['corrDenMat'],edSol['twoPart'],unCorrDenMatOrb,coeffsComplete,coeffsOrbComplete)
    #print('stiching',(time.time() - start)*1000.0)
    # <H - H*>*

    energyEff = energyEffUnCorr + edSol['energyEffCorr']
    PhiComplete = edSol['PhiCorr'] + PhiUnCorr

    diffEnergy = energy - energyEff
    
    phiTilde = PhiComplete + diffEnergy
    #print(phiTilde)
#    if len(pointX)>20:
#        exit()
    return phiTilde


def calcPhiTildeEDExplicitPoint(ham,cfg,par,parEff,blockCar,hFSol):
    #%% 
    #print 'iter start Radial'
    #print('second point print')
    
    # either get the result from the last iteration, which is in the current pickle-file
    # or make a first guess, which are simply flat \tilde V_i
    cfg['tempFileConstrain'] = cfg['algo']['scratchDir'] + 'con'+ cfg['dateString'] + '.pkl'
    if os.path.isfile(cfg['tempFileConstrain']):
        
        point, cD, cC= io.getRecentOnePartBasis(cfg)
        #cD[:,1:] =0.05* np.random.rand(cD.shape[1]-1)
        point = vecToPoint(cfg,par, cD, cC)
        # if current point is present, use it to transform ls terms
        if parEff.lsCoupling:
            print(cD)
    else:
        #print('making new')
        startCoeffsD, startCoeffsC1 = harFock.firstGuessHFSubSpace(par)
        point = vecToPoint(cfg, par, startCoeffsD, startCoeffsC1)

        # if no point is present, use original ls -terms
        if parEff.lsCoupling:
            print(startCoeffsD)
            parEff.lsMat = copy.copy(par.lsMat)
    start = time.time()
    
    print('solving ED:')
    edSol = funcEDPhi(cfg,par,parEff,blockCar,ham)

    print('ED finished after '+str(time.time()-start)+ ' sec')
    #print 'warning hacked code'
    #startCoeffsD, startCoeffsC1 = harFock.firstGuessHFSubSpace(par)
    #point = vecToPoint(par, startCoeffsD, startCoeffsC1)
    gOrbOrtho = dict()
    gOrbNormC = dict()
    gOrbNormD = dict()

    currentFixPoint = copy.copy(point)
    if par.numDeg == 0:
        lengthPoint = currentFixPoint.size/par.NimpOrbs
    else:
        lengthPoint = currentFixPoint.size/par.numDeg

    start = time.time()

    if par.numDeg == 0:
        for iO in range(par.NimpOrbs):
            if cfg['algo']['optOnlyBath']:
                gOrbNormC = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:],pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:].transpose()) - 1.0
                consOrb = [{'type' : 'eq', 'fun' : gOrbNormC}]
            else:
                gOrbOrtho = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[0][iO,:],pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:].transpose()) - 0.0
                gOrbNormC = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:],pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:].transpose()) - 1.0
                gOrbNormD = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[0][iO,:],pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[0][iO,:].transpose()) - 1.0
                consOrb = [{'type' : 'eq', 'fun' : gOrbOrtho},
                           {'type' : 'eq', 'fun' : gOrbNormC},
                           {'type' : 'eq', 'fun' : gOrbNormD}]

            o = optimize.minimize(calcEnergyHFFixEDOrb, x0=currentFixPoint[iO*lengthPoint:(iO+1)*lengthPoint], args=(par,hFSol,edSol,iO,currentFixPoint,cfg), constraints=consOrb, method='SLSQP')
            #print(o)
            currentFixPoint[iO*lengthPoint:(iO+1)*lengthPoint] = copy.copy(o.x)
            #print 'constrain optim for orb ' + str(iO)+' '+str(time.time() - start)+'s after',o.nfev,'iterations', o.fun, pointX 
    else:
        

        for iD in range(par.numDeg):
            if cfg['algo']['optOnlyBath']:
                gOrbNormC = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:],pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:].transpose()) - 1.0
                consOrb = ({'type' : 'eq', 'fun' : gOrbNormC})

            else:
                gOrbOrtho = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[0][par.degeneracy[iD][0],:],pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:].transpose()) - 0.0
                gOrbNormC = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:],pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:].transpose()) - 1.0
                gOrbNormD = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[0][par.degeneracy[iD][0],:],pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[0][par.degeneracy[iD][0],:].transpose()) - 1.0
                consOrb = ({'type' : 'eq', 'fun' : gOrbOrtho},
                           {'type' : 'eq', 'fun' : gOrbNormC},
                           {'type' : 'eq', 'fun' : gOrbNormD})
                listCons = (gOrbOrtho,gOrbOrtho,gOrbOrtho)
                #print(consOrb)
                #print('constrains')
                #for ii in range(3):
                #    print(consOrb[ii]['fun'](currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint]))
            #xMin,fMin,its,eMode,sMode = optimize.fmin_slsqp(calcEnergyHFFixEDOrb,currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint],listCons,args=(par,hFSol,edSol,iD,currentFixPoint))
            #_minimize_slsqp(fun, x0, args, jac, bounds,
            #                constraints, callback=callback, **options)
            #currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint] = copy.copy(xMin)
            #print('constrain optim for orb ' + str(iD)+' '+str(time.time() - start)+'s after',its,'iterations', fMin, pointX)
                
            o = optimize.minimize(calcEnergyHFFixEDOrb, x0=currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint], args=(par,hFSol,edSol,iD,currentFixPoint,cfg), constraints=consOrb, method='SLSQP')
            print('success',o.success,'with',o.message)
            currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint] = copy.copy(o.x)
            
            
            print('constrain optim for orb {0:d} in {1:1.2f} after {2:d} funcEvals {3:d}, iterations:'.format(iD,time.time() - start,o.nfev), o.fun,o.iter)
            

   
    print(o)           
    print('one particle optimization finished with PhiTilde=',o.fun)
    coeffsD, coeffsC = pointToVec(cfg,par,currentFixPoint)

    f = open(cfg['tempFileConstrain'],'wb')
    guess = dict()
    guess['point'] = currentFixPoint
    guess['coeffsD'] = copy.copy(coeffsD)
    guess['coeffsC'] = copy.copy(coeffsC)
    
    pickle.dump(guess,f)
    f.close()
    if False:
        plt.figure(2)
        plt.plot(par.epsBath[0,:],coeffsD[0,1:],'--xb')
        plt.plot(0,coeffsD[0,0],'xb')
        plt.plot(par.epsBath[0,:],coeffsC[0,1:],'--r')
        plt.plot(0,coeffsC[0,0],'xr')
        plt.title('linear combinations of HF eigenvectors')
        plt.show()
    
    return o.fun

def calcPhiTildeED(pointX,*args):
    #%% 
    #print 'iter start Radial'
    #print('second point print')
    print(pointX)
    ham = args[0]
    cfg = args[1]
    par = args[2]
    parEff = args[3]
    blockCar = args[4]
    hFSol = args[5]

    if par.NimpOrbs > 5:
        print('iter start')
    print()
    


    epsTilde, vBar, epsBath = varVectorToParams(pointX,cfg,par)

    # if we want to  vary U...
    #epsTilde, vBar, epsBath, U = varVectorToParamswU(pointX,par)
    #parEff.UImp = U
    #parEff.JImp = U/4.0
    #parEff.uMatrix = uMat.uMatrixWrapper(parEff)

    parEff.epsImp = epsTilde[:,0]
    parEff.vBath[:,0] = vBar[:,0]
    parEff.epsBath[:,0] = epsBath[:,0]
    

    

    # either get the result from the last iteration, which is in the current pickle-file
    # or make a first guess, which are simply flat \tilde V_i
    cfg['tempFileConstrain'] = cfg['algo']['scratchDir'] + 'con'+ cfg['dateString'] + '.pkl'
    if os.path.isfile(cfg['tempFileConstrain']):
        
        point, cD, cC= io.getRecentOnePartBasis(cfg)
        #cD[:,1:] =0.05* np.random.rand(cD.shape[1]-1)
        point = vecToPoint(cfg,par, cD, cC)
        # if current point is present, use it to transform ls terms
        if parEff.lsCoupling:
            print(cD)
        if cfg['ed']['excited'] != 0:
            currentFixPoint = copy.copy(point)
            if par.numDeg == 0:
                lengthPoint = currentFixPoint.size/par.NimpOrbs
            else:
                lengthPoint = currentFixPoint.size/par.numDeg
            edSol = funcEDPhi(cfg,par,parEff,blockCar,ham)
            return calcEnergyHFFixEDOrb(currentFixPoint[0*lengthPoint:(0+1)*lengthPoint],par,hFSol,edSol,0,point,cfg)
    else:
        #print('making new')
        startCoeffsD, startCoeffsC1 = harFock.firstGuessHFSubSpace(par)
        point = vecToPoint(cfg, par, startCoeffsD, startCoeffsC1)
        #print('first',point)
        #exit()
        # if no point is present, use original ls -terms
        if parEff.lsCoupling:
            print(startCoeffsD)
            parEff.lsMat = copy.copy(par.lsMat)
    start = time.time()
    print('solving ED:')
    edSol = funcEDPhi(cfg,par,parEff,blockCar,ham)

    print('ED finished after '+str(time.time()-start)+ ' sec')
    #print 'warning hacked code'
    #startCoeffsD, startCoeffsC1 = harFock.firstGuessHFSubSpace(par)
    #point = vecToPoint(par, startCoeffsD, startCoeffsC1)
    gOrbOrtho = dict()
    gOrbNormC = dict()
    gOrbNormD = dict()

    currentFixPoint = copy.copy(point)
    if par.numDeg == 0:
        lengthPoint = currentFixPoint.size/par.NimpOrbs
    else:
        lengthPoint = currentFixPoint.size/par.numDeg

    start = time.time()

    if par.numDeg == 0:
        for iO in range(par.NimpOrbs):
            if cfg['algo']['optOnlyBath']:
                gOrbNormC = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:],pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:].transpose()) - 1.0
                consOrb = [{'type' : 'eq', 'fun' : gOrbNormC}]
            else:
                gOrbOrtho = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[0][iO,:],pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:].transpose()) - 0.0
                gOrbNormC = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:],pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[1][iO,:].transpose()) - 1.0
                gOrbNormD = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[0][iO,:],pointToVecOrbStitch(cfg,par,x,iO,currentFixPoint)[0][iO,:].transpose()) - 1.0
                consOrb = [{'type' : 'eq', 'fun' : gOrbOrtho},
                           {'type' : 'eq', 'fun' : gOrbNormC},
                           {'type' : 'eq', 'fun' : gOrbNormD}]
            o = optimize.minimize(calcEnergyHFFixEDOrb, x0=currentFixPoint[iO*lengthPoint:(iO+1)*lengthPoint], args=(par,hFSol,edSol,iO,currentFixPoint,cfg), constraints=consOrb, method='SLSQP', options={'disp': False,'ftol' : cfg['algo']['deltaInner'],'maxiter':cfg['algo']['maxIterInner']})
            #print(o)
            currentFixPoint[iO*lengthPoint:(iO+1)*lengthPoint] = copy.copy(o.x)
            #print 'constrain optim for orb ' + str(iO)+' '+str(time.time() - start)+'s after',o.nfev,'iterations', o.fun, pointX 
    else:
        

        for iD in range(par.numDeg):
            if cfg['algo']['optOnlyBath']:
                gOrbNormC = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:],pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:].transpose()) - 1.0
                consOrb = ({'type' : 'eq', 'fun' : gOrbNormC})

            else:
                gOrbOrtho = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[0][par.degeneracy[iD][0],:],pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:].transpose()) - 0.0
                gOrbNormC = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:],pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[1][par.degeneracy[iD][0],:].transpose()) - 1.0
                gOrbNormD = lambda x: np.dot(pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[0][par.degeneracy[iD][0],:],pointToVecOrbStitch(cfg,par,x,iD,currentFixPoint)[0][par.degeneracy[iD][0],:].transpose()) - 1.0
                consOrb = ({'type' : 'eq', 'fun' : gOrbOrtho},
                           {'type' : 'eq', 'fun' : gOrbNormC},
                           {'type' : 'eq', 'fun' : gOrbNormD})
                listCons = (gOrbOrtho,gOrbOrtho,gOrbOrtho)
                #print(consOrb)
                #print('constrains')
                #for ii in range(3):
                #    print(consOrb[ii]['fun'](currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint]))
            #xMin,fMin,its,eMode,sMode = optimize.fmin_slsqp(calcEnergyHFFixEDOrb,currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint],listCons,args=(par,hFSol,edSol,iD,currentFixPoint))
            #_minimize_slsqp(fun, x0, args, jac, bounds,
            #                constraints, callback=callback, **options)
            #currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint] = copy.copy(xMin)
            #print('constrain optim for orb ' + str(iD)+' '+str(time.time() - start)+'s after',its,'iterations', fMin, pointX)
            o = optimize.minimize(calcEnergyHFFixEDOrb, x0=currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint], args=(par,hFSol,edSol,iD,currentFixPoint,cfg), constraints=consOrb, method='SLSQP', options={'disp': False,'ftol' : cfg['algo']['deltaInner'],'maxiter':cfg['algo']['maxIterInner']})
            print('success',o.success,'with',o.message)
            currentFixPoint[iD*lengthPoint:(iD+1)*lengthPoint] = copy.copy(o.x)
            print('constrain optim for orb {0:d} in {1:1.2f}s after {2:d} funcs, {3:d} iterations:'.format(iD,time.time() - start,o.nfev,o.nit), o.fun)
            

   
               
    print('one particle optimization finished with PhiTilde=',o.fun,pointX)
    coeffsD, coeffsC = pointToVec(cfg,par,currentFixPoint)
    if cfg['ed']['excited'] == 0:
        f = open(cfg['tempFileConstrain'],'wb')
        guess = dict()
        guess['point'] = currentFixPoint
        guess['coeffsD'] = copy.copy(coeffsD)
        guess['coeffsC'] = copy.copy(coeffsC)
        pickle.dump(guess,f)
        f.close()
    if False:
        plt.figure(2)
        plt.plot(par.epsBath[0,:],coeffsD[0,1:],'--xb')
        plt.plot(0,coeffsD[0,0],'xb')
        plt.plot(par.epsBath[0,:],coeffsC[0,1:],'--r')
        plt.plot(0,coeffsC[0,0],'xr')
        plt.title('linear combinations of HF eigenvectors')
        plt.show()
    # update HF solution
    #if cfg['hf']['updateUncorrelated']
    #    completeDenMatLocBasis, _, _, _, _, _, _ = hel.calcUpdatedObservables(o.x,ham,cfg,par,parEff,blockCar,hFSol)
    #    hFSol = updateHartreeFock(par,cfg,completeDenMatLocBasis)
    return o.fun


def opt(cfg,point,ham,par,parEff,blockCar,hfSol):
    #%%     
    #checkFirstGuess(point,par)
    parEff.twoPart = set_ham_pure_twoPart(parEff,ham)
    print('implement first guess checker!')
    if cfg['algo']['optOnlyBath']:
        print('optimizing only bath basis')
    else:
        print('optimizing complete basis')
    bfgsOpts = {'gtol' : cfg['algo']['deltaOuter']}
    NMOpts = {'xtol' : cfg['algo']['deltaOuter'],
              'ftol' : cfg['algo']['deltaOuter'],
              'maxfev': cfg['algo']['maxIterOuter']}
    powOpts = {'ftol': cfg['algo']['deltaOuter']}

    out=dict()
    # calculate two particle part of hamiltonian, this is unchanged in any case
    
    
    # optimizing    
    if cfg['algo']['maxIterOuter'] == 0:
        print('performing no outer loop! (no optimization of parameters of Hamiltonian)')
        out= dict()
        out['xopt'] = point               
        out['fopt'] = calcPhiTildeED(point,ham, cfg, par, parEff, blockCar,hfSol)
        out['func_calls'] = 0
        print('optimal PhiTilde', out['fopt'],'after only optimizing basis',out['xopt'])
    else:
        print(point)
        print("optimizing parameters of Hamiltonian (",len(point),"parameters)...",end=" ")
        o=optimize.minimize(calcPhiTildeED,point,args=(ham,cfg,par,parEff,blockCar,hfSol),method='Nelder-Mead',options=NMOpts)
        #o=optimize.minimize(calcPhiTildeED,point,args=(ham,par,parEff,spPars,blockCar,hfSol),method='BFGS',options=bfgsOpts)
       
        out['func_callsBFGS'] = o.nfev
        o.success = True
        if o.success == False:
            print('BFGS did not succeed. Using Nelder-Mead...')
            o=optimize.minimize(calcPhiTildeED,o.x,args=(ham,cfg,par,parEff,blockCar,hfSol),method='BFGS',options=NMOpts)
            out['func_callsNM'] = o.nfev
        else:
            out['func_callsNM'] = 0
    
        
        out['xopt'] = o.x
        out['fopt'] = o.fun
        out['func_calls'] = out['func_callsNM'] + out['func_callsBFGS']
        print('optimal PhiTilde', out['fopt'], 'number of BFGS steps:', out['func_callsBFGS'],'NM steps:', out['func_callsNM'],o.x)


    return out
