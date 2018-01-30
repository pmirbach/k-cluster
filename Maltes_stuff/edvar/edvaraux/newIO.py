# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:03:09 2014

@author: mschueler
"""
import numpy as np
#import parametersMulOrb
import time
import sys
import pickle
import os
import glob
from configobj import ConfigObj
from validate import Validator


def readHyb(params):
    hybData = np.genfromtxt(params['filename'])
    energy = hybData[:,0]
    
    Delta = hybData[:,2::2] - 1j*hybData[:,3::2]
    DeltaCut = None
    #DeltaCut = np.zeros(shape=(Delta.shape[0],len(params['onlyBands'])),dtype=complex)
    #for iB in range(len(params['onlyBands'])):
    #    DeltaCut[:,iB] = Delta[:,params['onlyBands'][iB]]
    return energy, Delta, DeltaCut
    
    
    
def checkInterpolate(epsTilde,vBar,parEff):
    fileNameData = 'N{0:d}/U{1:1.3f}J{2:1.3f}mu{3:1.3f}'.format(parEff.NimpOrbs,parEff.UImp,parEff.JImp,parEff.mu)
    if os.path.isfile(fileNameData):
        data = np.genfromtxt(fileNameData)

        if len(data.shape) > 1:
            epsData = data[:,0]
            vData = data[:,1]
        else:
            epsData = data[0][np.newaxis]
            vData = data[1][np.newaxis]
        
        points = np.array([epsData,vData])
        
    else:
        points = np.zeros(shape=(1,1))
    return points
        
def pickleData(epsTilde,vBar,den1,cd,Phi1,E0,parEff):
    pickDic = dict()
    pickDic['epsT'] = epsTilde
    pickDic['vBar'] = vBar
    pickDic['den'] = den1
    pickDic['cd'] = cd
    pickDic['Phi1'] = Phi1
    pickDic['E0'] = E0
    directory = 'N{0:d}/'.format(parEff.NimpOrbs)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fileName = 'N{0:d}/U{1:1.3f}J{2:1.3f}mu{3:1.3f}e{4:1.8f}v{5:1.8f}'.format(parEff.NimpOrbs,parEff.UImp,parEff.JImp,parEff.mu,epsTilde[0,0],vBar[0])
    f = open(fileName,'wb')
    pickle.dump(pickDic,f)
    f.close()
    
    fileNameData = 'N{0:d}/U{1:1.3f}J{2:1.3f}mu{3:1.3f}'.format(parEff.NimpOrbs,parEff.UImp,parEff.JImp,parEff.mu)
    
    if os.path.isfile(fileNameData):
        fD = open(fileNameData,'a')
        fD.write('{0:1.16e} {1:1.16e}\n'.format(epsTilde[0,0],vBar[0]))
        fD.close()
    else:
        fD = open(fileNameData,'w')
        fD.write('{0:1.16e} {1:1.16e}\n'.format(epsTilde[0,0],vBar[0]))
        fD.close()


    
    

def unpickleData(epsTilde,vBar,parEff):
    fileName = 'N{0:d}/U{1:1.3f}J{2:1.3f}mu{3:1.3f}e{4:1.8f}v{5:1.8f}'.format(parEff.NimpOrbs,parEff.UImp,parEff.JImp,parEff.mu,epsTilde[0,0],vBar[0])
    f = open(fileName,'rb')
    pickDic = pickle.load(f)
    f.close()
    return pickDic['den'],pickDic['cd'],pickDic['Phi1'],pickDic['E0']
    


def saveResultNew(cfg,out):
    if cfg['algo']['save']:
        # write to ascii file
        if cfg['algo']['appendDateString']:
            dateString = cfg['dateString']
        else:
            dateString = ''
        print('saving to', cfg['algo']['resultFileName']+dateString+'.pkl')
        #parReal = parametersMulOrb.parametersrealMat()
        #spPars = parametersMulOrb.sparseParameters() 
        
        f = open(cfg['algo']['resultFileName']+dateString+'.pkl','wb')
#        outPickle = {'parameters' : par,
#                     'parHF':parHF,
#                     'results' : out}
        outPickle = {'parameters' : cfg,
                     'results' : out}

        pickle.dump(outPickle,f)
        f.close()
        

def readConfig(filename=None):
    if filename == None:
        filename = 'parameters.in'
    pa = sys.path
    for direc in pa:
        dummy = glob.glob(direc+'/edvaraux/configspec')
        if len(dummy) == 1:
            cfgSpecFileName = dummy[0]
    #f = open(cfgSpecFileName,'r')
    #print(f.readlines())
    configspec = ConfigObj(cfgSpecFileName,_inspec=True)

    
    validator = Validator()
    
    cfg = ConfigObj(filename, configspec=configspec)
    cfg.validate(validator)
    return cfg
    
def getRecentCon(cfg):
    f = open(cfg['tempFileConstrain'],'rb')
    pointV = pickle.load(f)
    f.close()
    return pointV
    
def getRecentOnePartBasis(cfg):
    f = open(cfg['tempFileConstrain'],'rb')
    pickDic = pickle.load(f)
    f.close()
    #for iO in range(par.NimpOrbs):
    #    pickDic['coeffsD'] /= np.sqrt(np.sum(pickDic['coeffsD']**2))
    #    pickDic['coeffsC'] /= np.sqrt(np.sum(pickDic['coeffsC']**2))
    return pickDic['point'], pickDic['coeffsD'], pickDic['coeffsC']

def saveEDSolution(cfg,edSol):
    if cfg['ed']['excited'] == 1:
        fileName = cfg['algo']['scratchDir'] + 'eigenVecs/edSolution+1' + cfg['dateString']
    elif cfg['ed']['excited'] == -1:
        fileName = cfg['algo']['scratchDir'] + 'eigenVecs/edSolution-1' + cfg['dateString']
    elif cfg['ed']['excited'] == 0:
        fileName = cfg['algo']['scratchDir'] + 'eigenVecs/edSolution' + cfg['dateString']
            
    print("saving ED solution to",fileName)
    f = open(fileName,'wb')
    pickle.dump(edSol,f)
    f.close()
    return

    

def fetchLastEDSolution(cfg):
    if cfg['ed']['excited'] == 1:
        fileName = cfg['algo']['scratchDir'] + 'eigenVecs/edSolution+1' + cfg['dateString']
    elif cfg['ed']['excited'] == -1:
        fileName = cfg['algo']['scratchDir'] + 'eigenVecs/edSolution-1' + cfg['dateString']
    elif cfg['ed']['excited'] == 0:
        fileName = cfg['algo']['scratchDir'] + 'eigenVecs/edSolution' + cfg['dateString']
    #fileName = cfg['algo']['scratchDir'] + 'eigenVecs/edSolution' + cfg['dateString']
    print('loading last ED solutionfrom',fileName)
    f = open(fileName,'rb')
    edSol = pickle.load(f)
    f.close()
    return edSol

def readPickle(filename):
    f = open(filename,'rb')
    inPickle = pickle.load(f)
    f.close()
    return inPickle['parameters'], inPickle['results']
    
def readResults(filename):
    f = open(filename,'r')
    a=f.readlines()
    valStr = []
    for iL in range(len(a)):
        ind = a[iL].find(':')
        valStr.append(a[iL][ind+1:-1])
    out = dict()
    realMat = bool(valStr[0])
    realMatepsMin = float(valStr[1])
    realMatepsMax = float(valStr[2])
    realMatonlyBands = float(valStr[3])
    sparseNthreshhold = int(valStr[4])
    sparseNvalues = int(valStr[5])
    legendre = bool(valStr[6])
    legendreOrder = int(valStr[7])
    groundstate = bool(valStr[8])
    beta = float(valStr[9])
    useUnCorrProjEig = bool(valStr[10])
    degeneracyNumDeg = int(valStr[11])
    print(valStr[12])
    degeneracyList = 0
    degeneracyOverRide = bool(valStr[13])
    NimpOrbs = int(valStr[14])
    bathsites = int(valStr[15])
    UImp = float(valStr[16])
    JImp = float(valStr[17])
    uMatonlyBands = int(valStr[18])
    epsImp = [float(ii) for ii in valStr[19].split()]
    mu = float(valStr[20])
    PhiTilde0 = float(valStr[21])
    funcCallsForMin = int(valStr[22])
    xopt = [float(ii) for ii in valStr[23].split()]
    
def checkInput(par,parEff,cfg):
    

    
    if par.NimpOrbs > 1 and cfg['algo']['optU']:
        raise Exception('not implemented variation of U for NimpOrbs > 1')
    
    if par.numDeg == 0 or parEff.numDeg == 0:    
        raise Exception('parameter numDeg = 0 depreciated')
    
    print('checking input',cfg['aim']['overRideV_Deg_Check'])
    if (par.vBath.shape[1] != par.Nbath) or par.vBath.shape[0] != par.NimpOrbs:
        raise Exception('vbath dimension does not fit to number of baths or orbitals')
    if (par.epsBath.shape[1] != par.Nbath) or par.epsBath.shape[0] != par.NimpOrbs:
        raise Exception('epsbath dimension does not fit to number of baths or orbitals')
    if (par.epsImp.size != par.NimpOrbs) or par.epsBath.shape[0] != par.NimpOrbs:
        raise Exception('epsImp dimension does not fit to number of baths or orbitals')
    
    if cfg['algo']['fitLegendre']:
        if cfg['algo']['legendreOrder'] < 1:
            raise Exception('legendreOrder is 0, but has to be >0')
    try:
        toleranceVCheck=cfg['aim']['degeneracyTol']
    except:
        toleranceVCheck = 1e-15


    if (par.numDeg > 0):
        if (par.numDeg != len(par.degeneracy)):
            raise Exception('number of degeneracies given in list par.degeneracy does not fit not number in par.numDeg')
        # check if mean over orbital energy and hyb is equal to all energies and hybs
        for iD in range(par.numDeg):
            meanVals = np.mean(par.epsBath[par.degeneracy[iD],:],axis=0)
            for iOD in range(len(par.degeneracy[iD])):
                if np.any(np.abs(meanVals - par.epsBath[par.degeneracy[iD][iOD],:])>1e-15):
                    raise Exception('degenercy of energies not recognized, this check should NOT be overriden')
            meanVals = np.mean(par.vBath[par.degeneracy[iD],:],axis=0)
            if cfg['aim']['overRideV_Deg_Check'] == False:
                for iOD in range(len(par.degeneracy[iD])):    
                    if np.any(np.abs(meanVals - par.vBath[par.degeneracy[iD][iOD],:])>toleranceVCheck):
                        maxDiscrep = np.max(np.abs(meanVals - par.vBath[par.degeneracy[iD][iOD],:]))
                        print('maximal discrepeancy from mean value:',maxDiscrep)
                        raise Exception('degenercy of hyb not recognized, maybe override this check..')
                # to really have degenerate bands, we now use the mean values as input:
                par.vBath[par.degeneracy[iD][:],:] = meanVals
                meanVals = np.mean(par.epsImp[par.degeneracy[iD]])
                par.epsImp[par.degeneracy[iD][:]] = meanVals
                par.epsImp[parEff.degeneracy[iD][:]] = meanVals
            else:
                print('Warning: forcing requested degeneracies by using mean of hybridizations')
                # to really have degenerate bands, we now use the mean values as input:
                par.vBath[par.degeneracy[iD][:],:] = meanVals
                meanVals = np.mean(par.epsImp[par.degeneracy[iD]])
                par.epsImp[par.degeneracy[iD][:]] = meanVals
                par.epsImp[parEff.degeneracy[iD][:]] = meanVals
    return par
