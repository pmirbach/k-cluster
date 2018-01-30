# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:08:27 2014

@author: mschueler
"""
import numpy as np
import scipy.misc as misc

class matrices:
    def __init__(self):
        # clebsh gordon coefficients for l=2, s=1/2
        # from pdg.lbl.gov/2002/clebrpp.pdf
    
        # first index is j,mj [5/2 5/2;5/2 3/2;3/2 3/2;5/2 1/2;3/2 1/2;5/2 -1/2;3/2 -1/2;5/2 -3/2;3/2 -3/2;5/2 -5/2]
        # second index is ml,ms [-2 1/2; -1 1/2; 0 1/2; 1 1/2; 2 1/2; -2 -1/2; -1 -1/2; 0 -1/2; 1 -1/2; 2 -1/2]
        self.jVec = np.array([5,5,3,5,3,5,3,5,3,5])/2.0
        self.mjVec = np.array([5,3,3,1,1,-1,-1,-3,-3,-5])/2.0
        self.mlVec = np.array([-2,-1,0,1,2,-2,-1,0,1,2])
        self.msVec = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])/2.0
        self.clebMat = np.zeros((10,10))
        
        # 5/2 5/2
        self.clebMat[0,4] = 1.0
        # 5/2 3/2
        self.clebMat[1,9] = 1.0/5.0
        self.clebMat[1,3] = 4.0/5.0
        # 3/2 3/2
        self.clebMat[2,9] = 4.0/5.0
        self.clebMat[2,3] =-1.0/5.0
        # 5/2 1/2
        self.clebMat[3,8] = 2.0/5.0
        self.clebMat[3,2] = 3.0/5.0
        # 3/2 1/2
        self.clebMat[4,8] = 3.0/5.0
        self.clebMat[4,2] =-2.0/5.0
        # 5/2 -1/2
        self.clebMat[5,7] = 3.0/5.0
        self.clebMat[5,1] = 2.0/5.0
        # 3/2 -1/2
        self.clebMat[6,7] = 2.0/5.0
        self.clebMat[6,1] =-3.0/5.0
        # 5/2 -3/2
        self.clebMat[7,6] = 4.0/5.0
        self.clebMat[7,0] = 1.0/5.0
        # ;
        self.clebMat[8,6] = 1.0/5.0
        self.clebMat[8,0] =-4.0/5.0
        # 5/2 -5/2
        self.clebMat[9,5] = 1.0
        
        self.clebMat[self.clebMat>0] = np.sqrt(self.clebMat[self.clebMat>0])
        self.clebMat[self.clebMat<0] = -np.sqrt(np.abs(self.clebMat[self.clebMat<0]))
        
        
        self.rotMat = np.zeros(shape=(5,5),dtype=complex)
        sq2 = 1/np.sqrt(2)
        self.rotMat[0,0] = 1j*sq2
        self.rotMat[0,4] =-1j*sq2
        self.rotMat[1,1] = 1j*sq2
        self.rotMat[1,3] = 1j*sq2
        self.rotMat[2,2] = 1
        self.rotMat[3,1] = sq2
        self.rotMat[3,3] =-sq2
        self.rotMat[4,0] = sq2
        self.rotMat[4,4] = sq2


def lsCouplingL2Spherical():
    
    cMat = matrices().clebMat
    aMat = np.linalg.inv(cMat)
    
    jVec = matrices().jVec
    
    # transform fom |j mj> to |ml ms>
    jMat = np.zeros((10,10))
    for i1 in range(10):
        for i2 in range(10):
            # combined sum over j and mj
            for jj in range(10):
                jMat[i1,i2] += jVec[jj]*(jVec[jj]+1.0) * aMat[i1,jj] * aMat[i2,jj]
                
    
    lMat = 6.0*np.eye(10)
    sMat = 3.0/4.0*np.eye(10)
    
    lsMat = 0.5*(jMat - lMat - sMat)
    lsMat[np.abs(lsMat)<1e-14] = 0    
    
    return lsMat
    
    
def lsCouplingL2Cubic():
    
    lsMatSpher = lsCouplingL2Spherical()
    rotMat = matrices().rotMat
    rotMatSpin = np.zeros((10,10),dtype=complex)
    rotMatSpin[:5,:5] = rotMat
    rotMatSpin[5:10,5:10] = rotMat
    #lsMatCub = np.dot(np.dot(rotMatSpin,lsMatSpher),rotMatSpin.conjugate().transpose())
    #print np.all(np.abs(lsMatCub - lsMatCub.transpose().conjugate())<1e-14)
    
    lsMatCub  =np.zeros(shape=(10,10),dtype=complex)
    for ii in range(10):
        for jj in range(10):
            dum = np.tensordot(np.conj(rotMatSpin[ii,:]),lsMatSpher,axes=(0,0))
            lsMatCub[ii,jj] = np.tensordot(dum,rotMatSpin[jj,:],axes=(0,0))
    #print np.all(np.abs(lsMatCub - lsMatCub.transpose().conjugate())<1e-14)
    #print lsMatCub2 - lsMatCub
    #a=b
                    
    return lsMatCub


def matrixWrapper(par):
    # sz

    ll = (par.NimpOrbs-1)/2
    lz = np.zeros((par.NimpOrbs,par.NimpOrbs))
    lp = np.zeros((par.NimpOrbs,par.NimpOrbs))
    lm = np.zeros((par.NimpOrbs,par.NimpOrbs))
    
    for iO in range(par.NimpOrbs):
        m = -ll+iO
        lz[iO,iO] = m
    for iO in range(par.NimpOrbs-1):
        m = -ll+iO
        matEle = np.sqrt((ll-m)*(ll+m+1)*0.5)
        lp[iO+1,iO] = matEle
    for iO in range(1,par.NimpOrbs):
        m = -ll+iO
        matEle = np.sqrt((ll+m)*(ll-m+1)*0.5)
        lm[iO-1,iO] = matEle
      
    # rotate into cubic basis
    rotMat = matrices().rotMat
    lzR = np.zeros(lz.shape,dtype=complex)
    lpR = np.zeros(lz.shape,dtype=complex)
    lmR = np.zeros(lz.shape,dtype=complex)
    for ii in range(5):
        for ll in range(5):
            dum1 = np.tensordot(np.conj(rotMat[ii,:]),lz,axes=(0,0))
            lzR[ii,ll] = np.tensordot(dum1,rotMat[ll,:],axes=(0,0))
            dum1 = np.tensordot(np.conj(rotMat[ii,:]),lp,axes=(0,0))
            lpR[ii,ll] = np.tensordot(dum1,rotMat[ll,:],axes=(0,0))
            dum1 = np.tensordot(np.conj(rotMat[ii,:]),lm,axes=(0,0))
            lmR[ii,ll] = np.tensordot(dum1,rotMat[ll,:],axes=(0,0))

    # something wrong here!  
    lzR = np.zeros(lz.shape,dtype=complex)
    lpR = np.zeros(lz.shape,dtype=complex)
    lmR = np.zeros(lz.shape,dtype=complex) 
    for i1 in range(5):
        for j1 in range(5):         
            for i2 in range(5):
                for j2 in range(5):
                    lzR[i1,j1] += rotMat[i2,i1].conjugate()*rotMat[j2,j1]*lz[i2,j2]
                    lpR[i1,j1] += rotMat[i2,i1].conjugate()*rotMat[j2,j1]*lp[i2,j2]
                    lmR[i1,j1] += rotMat[i2,i1].conjugate()*rotMat[j2,j1]*lm[i2,j2]
#    #lzR = np.dot(rotMat.conjugate().transpose(),np.dot(lz,rotMat)).real
    #lpR = np.dot(rotMat.conjugate().transpose(),np.dot(lp,rotMat))
    #lmR = np.dot(rotMat.conjugate().transpose(),np.dot(lm,rotMat))
            
    return lzR,lpR,lmR
    
def uMatrixWrapper(par,cfg):
    if par.NimpOrbs == 3:
        uMatrix = uMatrix3Band(par.UImp,par.JImp)
    
    elif par.NimpOrbs == 1:
        uMatrix = np.array([[[[par.UImp]]]])
    else:
        uMatrix = uMatrix5Band(par.UImp,par.JImp,0,None)
    # sparsify the umatrix:
    for ii in range(par.NimpOrbs):
        for jj in range(par.NimpOrbs):
            for kk in range(par.NimpOrbs):
                for ll in range(par.NimpOrbs):
                    if np.abs(uMatrix[ii,jj,kk,ll])<1e-14:
                        uMatrix[ii,jj,kk,ll] = 0.0
    return uMatrix
# stuff for the calculation of the u-matrix                                     
        
def clebschGordan(j1,m1,j2,m2,j,m):

    cleGor = 0.

    if ( (abs(j1+j2) < j ) | (j < ( abs(j1-j2) )) ):
        return cleGor
    if ( abs(m1+m2-m) > 0 ):
        return cleGor
    if( (j1+j2+j)%1 > 1e-14 ):
        return cleGor
    if ( j1 < 0 | j2 < 0 ):
        return cleGor
    if (  (abs(m1) > j1) | (abs(m2) > j2 )):
        return cleGor

    coef = ( (2*j+1) * misc.factorial(1+j+j1+j2) / misc.factorial(j1+j2-j)
                       / misc.factorial(j1-j2+j) / misc.factorial(-j1+j2+j) )
    coef = ( coef * misc.factorial(j+m) * misc.factorial(j-m)
                *misc.factorial(j1+m1) * misc.factorial(j1-m1) )
    coef = coef / misc.factorial(j2+m2) / misc.factorial(j2-m2)
    coef = np.sqrt(coef)

    if (j1-m1)%2 == 1:
        coef = -coef

    a = 0
    si = 1
    for k in range(min(j-m,j1-m1)+1):
        a = a + (si*misc.factorial(j1+j2-m-k)*misc.factorial(j+j2-m1-k) /
                 (misc.factorial(k)) / (misc.factorial(j1-m1-k)) /
                 (misc.factorial(j-m-k)) / (misc.factorial(1+j+j1+j2-k)))
        si = -si
    cleGor = coef * a

    return cleGor

def aFak(k, l, m1, m2, m3, m4):
    a = 0.

    for q in range(-k,k+1):
        a += ( clebschGordan(l, m3, k, q, l, m1)
            *  clebschGordan(l, m2, k, q, l, m4) )

    a = a * clebschGordan( l,0, k,0, l,0 )**2
    return a

def uMatrix3Band(U,J):
    print('setting up 3 band u-matrix')
    nBands = 3
    l = int(( nBands - 1 )/2)


    uu = np.zeros(shape=(nBands,nBands,nBands,nBands),dtype=complex)
    FSlater = np.zeros(shape=(3),dtype=float)

    FSlater[0] = U
    FSlater[2] = 5*J


    uu = np.zeros(shape=(nBands,nBands,nBands,nBands),dtype=complex)
    fullUMat = np.zeros(shape=(nBands,nBands,nBands,nBands))
    mList = np.arange(-l,l+1)
    for m1 in mList:
        for m2 in mList:
            for m3 in mList:
                for m4 in mList:
                    # +l to fix the indexing   
                    fullUMat[m1+l,m2+l,m3+l,m4+l]= (
                                    FSlater[0] * aFak(0,l,m1,m2,m3,m4)
                                  + FSlater[2] * aFak(2,l,m1,m2,m3,m4)
                                                    )
    rotMat = np.zeros(shape=(3,3),dtype=complex)
    sq2 = 1.0/np.sqrt(2)
    rotMat[0,0] = 1j*sq2
    rotMat[0,2] = 1j*sq2
    rotMat[1,1] = 1.0
    rotMat[2,0] = sq2
    rotMat[2,2] = -sq2

    for ii in range(nBands):
        for jj in range(nBands):
            for kk in range(nBands):
                for ll in range(nBands):
                    dum1 = np.tensordot(np.conj(rotMat[ii,:]),fullUMat,axes=(0,0))
                    dum2 = np.tensordot(np.conj(rotMat[jj,:]),dum1,axes=(0,0))
                    dum3 = np.tensordot(dum2,rotMat[kk,:],axes=(0,0))
                    uu[ii,jj,kk,ll] = np.tensordot(dum3,rotMat[ll,:],axes=(0,0))
    return np.real(uu)


    
def uMatrix5Band(U,J,reduceMatrix,reducedBands):
    print('setting up 5 band u-matrix')
    nBands = 5
    l = int(( nBands - 1 )/2)

    if (reduceMatrix != 0) and (reduceMatrix != 1):
        raise Exception('set reduceMatrix to 0 or 1')

    # give a list of the bands for which reduced-matrix is written: 
    # order: xy, yz, z^2, xz, x^2; count from zero! 
    # example:                                                          
    # eg: [2, 4]                                                              
    # t2g: [0, 1, 3]                                                          
    # reducedBands = [0, 1, 3]                                                

    uu = np.zeros(shape=(nBands,nBands,nBands,nBands),dtype=complex)
    FSlater = np.zeros(shape=(5),dtype=float)

    FSlater[0] = U
    FSlater[2] = 14.*J/1.625
    FSlater[4] = 0.625 * FSlater[2]


    fullUMat = np.zeros(shape=(nBands,nBands,nBands,nBands))

    #mList = np.arange(-l,l+1)

    mList = range(-l,l+1)
    for m1 in mList:
        for m2 in mList:
            for m3 in mList:
                for m4 in mList:
                    # +l to fix the indexing   
                    fullUMat[m1+l,m2+l,m3+l,m4+l]= (
                                    FSlater[0] * aFak(0,l,m1,m2,m3,m4)
                                  + FSlater[2] * aFak(2,l,m1,m2,m3,m4)
                                  + FSlater[4] * aFak(4,l,m1,m2,m3,m4)
                                                    )
    rotMat = matrices().rotMat


#    for i1 in range(5):
#        for j1 in range(5):
#            for k1 in range(5):
#                for l1 in range(5): 
#                    for i2 in range(5):
#                        for j2 in range(5):
#                            for k2 in range(5):
#                                for l2 in range(5):
#            
#                                    uu[i1,j1,k1,l1] += rotMat[i2,i1].conjugate()*rotMat[j2,j1].conjugate()*rotMat[k2,k1]*rotMat[l2,l1]*fullUMat[i2,j2,k2,l2]


    for ii in range(nBands):
        for jj in range(nBands):
            for kk in range(nBands):
                for ll in range(nBands):
                    dum1 = np.tensordot(np.conj(rotMat[ii,:]),fullUMat,axes=(0,0))
                    dum2 = np.tensordot(np.conj(rotMat[jj,:]),dum1,axes=(0,0))
                    dum3 = np.tensordot(dum2,rotMat[kk,:],axes=(0,0))
                    uu[ii,jj,kk,ll] = np.tensordot(dum3,rotMat[ll,:],axes=(0,0))

    if reduceMatrix == 1:
        orbString = np.array(['xy','yz','z2','xz','x2'])
        print('used orbitals:', orbString[reducedBands])
        nBandsRed = len(reducedBands)
        uuRed = np.zeros(shape=(nBandsRed,nBandsRed,nBandsRed,nBandsRed),dtype=complex)
        for ii in range(nBandsRed):
            for jj in range(nBandsRed):
                for kk in range(nBandsRed):
                    for ll in range(nBandsRed):
                        uuRed[ii,jj,kk,ll] = uu[reducedBands[ii],reducedBands[jj],reducedBands[kk],reducedBands[ll]]
        uu = uuRed

    return np.real(uu)
