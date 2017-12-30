# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 02:22:13 2016

@author: Chris
"""
import math
from rectSegment import *
from numba import cuda
import numpy as np
from CudaKernels2d import *


class WaveSolver2D:
    
    def __init__(self,xdomains, ydomains,f,g,c,dt,bcs):
        self.beta_ = 2
        self.bcType_ = bcs

        self.xLength_ = xdomains[-1][-1]-xdomains[0][0]
        self.yLength_ = ydomains[-1][-1]-ydomains[0][0]

        self.dt_     = dt

        self.minSize_ = len(xdomains[0])
        
        for xdomain in xdomains:
            dsize = len(xdomain)
            
            if dsize<self.minSize_:
                self.minSize_ = dsize

        for ydomain in ydomains:
            dsize = len(ydomain)
            
            if dsize<self.minSize_:
                self.minSize_ = dsize

        self.xSegs_ = self.createDomains(xdomains)
        self.ySegs_ = self.createDomains(ydomains)

        if len(self.xSegs_) != len(self.ySegs_):
            assert("must be same number of x and y N points")

        # these parameters are used in calculating the values at the boundary
        self.alpha_ = self.beta_/(c*self.dt_)
        
        self.xdamping_ = math.exp(-self.alpha_*(self.xLength_))
        self.ydamping_ = math.exp(-self.alpha_*(self.yLength_))
        
        
        self.nDomains_ = len(self.xSegs_)
        self.time_ = self.dt_*2
        
        self.numThreads_ = self.minSize_

        #need a block for each subsegment, and then blocks for each y N point
        #block.x is the y or x point, There should be "N" of them
        #block.y is the subdomain being solved on the GPU
        self.numBlocks_  = (self.minSize_*self.nDomains_,self.nDomains_)
        
        
        # for ease of indexing, we are going to have a tuple as the key
        # I.E. rectSegments[(0,1)] will give the 0th x segment and 1st y segment
        self.rectSegments = {}
        for xind,xseg in enumerate(self.xSegs_):
            for yind,yseg in enumerate(self.ySegs_):
                self.rectSegments[(xind,yind)] = rectSegment(xseg,yseg,f,g,c,self.dt_)
        
        
    
    def createDomains(self,domains):
        endDomains = []
        #NOTE current behavior is to split and resize domains(upwards) until 
        #we have equal sized domains 
        for domain in domains:
            dsize = len(domain)

            if dsize != self.minSize_:
                newDomains = self.splitDomain(domain,dsize)
                
                for newDomain in newDomains:
                    endDomains.append(newDomain)

            else:
                endDomains.append(domain)
        return endDomains
    
    """
    waveHandler should know about the BC's, segments shouldn't have any idea
    what the boundary conditions are, this will set the correct bc conditions
    """
    def applyBCs(self, leftEnds , rightEnds):
        if self.bcType_ == 'periodic':
            self.BC1_ = leftEnds/(1-self.damping_)
            self.BC2_ = rightEnds/(1-self.damping_)
    
    """
    if the domain is larger than the minimum size, then we want to split it up and resize it
    until we have some number of domains that is each the minimum size
    this is done for load balancing on the gpu
    """   
    def splitDomain(self,xdomain,dsize):
        #get the beginning and end of the domain
        a     = xdomain[0]
        b     = xdomain[-1]
        
        if dsize%self.minSize_ != 0:
            print("domain not integer multiple, resizing")
            dsize = dsize + dsize%self.minSize_
        
        nDomains = int(dsize/self.minSize_) 
        stepSize = (b-a)/nDomains
        
        newDomains = [np.linspace(a+(i)*stepSize,a+(i+1)*stepSize,self.minSize_,dtype = np.float32) for i in range(nDomains)]
        return(newDomains)
    
    def timeStep(self):
        self.time_ += self.dt_
        
        empty3dMatrix    = np.zeros((self.numBlocks_[0],self.numBlocks_[1] , self.minSize_ ),dtype = np.float32)
        emptyMatrix      = np.zeros((self.numBlocks_[0],self.numBlocks_[1]),dtype=np.float32)

        hUValsMatrix      = empty3dMatrix
        hUPrevMatrix      = empty3dMatrix
        hPlist            = emptyMatrix
        hQlist            = emptyMatrix
        hRlist            = emptyMatrix
        hvjInvList        = emptyMatrix
        hdjlist           = emptyMatrix
        hquadCoeffsMatrix = empty3dMatrix
        hJLvalMatrix      = empty3dMatrix
        hJRvalMatrix      = empty3dMatrix
        hwMatrix          = empty3dMatrix
        hleftComm         = emptyMatrix
        hrightComm        = emptyMatrix
        hJSize            = np.uint32(self.minSize_)
        
        for subDomain in range(self.nDomains_):
            for subDomain2 in range(self.nDomains_):
                segment = self.rectSegments[(subDomain,subDomain2)]
                
                xindex1 = subDomain2*self.minSize_
                xindex2 =  (subDomain2+1)*self.minSize_

                yindex1 = subDomain
                yindex2 = (subDomain+1)
                print(self.numBlocks_)
                print(xindex1,xindex2,yindex1,yindex2)
                hUValsMatrix[xindex1:xindex2,yindex1:yindex2,:] = segment.ucurr_.reshape(self.minSize_,1,self.minSize_) 
                hUPrevMatrix[xindex1:xindex2,yindex1:yindex2,:] = segment.uprev_.reshape(self.minSize_,1,self.minSize_) 

                hPlist[xindex1:xindex2,yindex1:yindex2] = segment.getP('x')
                hQlist[xindex1:xindex2,yindex1:yindex2] = segment.getQ('x')
                hRlist[xindex1:xindex2,yindex1:yindex2] = segment.getR('x')
        
                hvjInvList[xindex1:xindex2,yindex1:yindex2] = np.float32(1/segment.getVj('x')**2)
                hdjlist[xindex1:xindex2,yindex1:yindex2]    = segment.getDj('x')
               
        # transfer all the data to the device
        
        dUValsMatrix  = cuda.to_device(hUValsMatrix)
        dPlist = cuda.to_device(hPlist)
        dQlist = cuda.to_device(hQlist)
        dRlist = cuda.to_device(hRlist)
        
        dvjInvList = cuda.to_device(hvjInvList)
        ddjlist    = cuda.to_device(hdjlist)

        dquadCoeffsMatrix = cuda.to_device(hquadCoeffsMatrix)
        dJLvalMatrix = cuda.to_device(hJLvalMatrix)
        dJRvalMatrix = cuda.to_device(hJRvalMatrix)

        dwMatrix    = cuda.to_device(hwMatrix)
        dleftComm   = cuda.to_device(hleftComm)
        drightComm  = cuda.to_device(hrightComm)
        
        dJSizeList  = hJSize
        
        """(type(dUValsMatrix), type(dPlist) 
             , type(dQlist)      , type(dRlist)
             , type(dvjInvList)        , type(ddjlist) 
             , type(dquadCoeffsMatrix) , type(dJLvalMatrix) 
             , type(dJRvalMatrix)      , type(dwMatrix) 
             , type(dleftComm)         , type(drightComm) 
             , type(dJSizeList))
        
        # we are going to do an x sweep first
        uLocal2D[self.numBlocks_,self.numThreads_](dUValsMatrix, dPlist 
             , dQlist            , dRlist
             , dvjInvList        , ddjlist 
             , dquadCoeffsMatrix , dJLvalMatrix 
             , dJRvalMatrix      , dwMatrix 
             , dleftComm         , drightComm 
             , dJSizeList) """
        
        ##NOTE All of the following values were written for convenience, 
        #      if it's a performance issue, put everything into 1 loop
        hxvals_ = emptyMatrix#numpy.zeros_like(hUvalsMatrix,np.dtype=np.float32)  
        ha_     = emptyMatrix
        hb_     = emptyMatrix
        
        for subDomain in range(self.nDomains_):
            yindex1 = subDomain
            yindex2 = (subDomain+1)
            
            #x grids are the same for every y currently
            segment = self.rectSegments[(subDomain,0)]
            hxvals_[:,yindex1:yindex2] = segment.x_

            ha_[:,yindex1:yindex2]     = segment.xa_
            hb_[:,yindex1:yindex2]     = segment.xb_
 
        self.hxvals_= hxvals_

        self.hleftComm  = dleftComm.copy_to_host()
        self.hrightComm = drightComm.copy_to_host()
        
        self.hA_ = emptyMatrix
        self.hB_ = emptyMatrix
        
        #try to cudaIze this, in 1d it doesn't make sense but in 2d it does 
        for i in range(self.numBlocks[0]):
            self.updateTransmissionCoeffs(i,self.leftComm[i,:],self.rightComm[i,:])

        dA_ = cuda.to_device(self.hA_)
        dB_ = cuda.to_device(self.hB_)

        da_          = cuda.to_device(ha_)
        db_          = cuda.to_device(hb_)
        dxvals_      = cuda.to_device(hxvals_)
        dUPrevMatrix = cuda.to_device (hUPrevMatrix)
        
        uUpdate2D[self.numBLocks_,self.numThreads_](dwMatrix, dxvals_ , dA_,dB_,self.alpha_ , da_, db_)
        
        for subDomain in range(nDomains):
            for subDomain2 in range(nDomains):
                segment = self.rectSegments[(subDomain2,subDomain)]
                
                xindex1 = subDomain2*self.minSize_
                xindex2 =  (subDomain2+1)*self.minSize_

                yindex1 = subDomain*self.nDomains_
                yindex2 = (subdomain+1)*self.nDomains_


                hPlist[xindex1:xindex2,yindex1:yindex2] = segment.getP('y')
                hQlist[xindex1:xindex2,yindex1:yindex2] = segment.getQ('y')
                hRlist[xindex1:xindex2,yindex1:yindex2] = segment.getR('y')
        
                hvjInvList[xindex1:xindex2,yindex1:yindex2] = 1/segment.getVj('y')**2
                hdjlist[xindex1:xindex2,yindex1:yindex2]    = segment.getDj('y')
        
        
        hwMatrix = dwMatrix.copy_to_host()
        hwMatrix = hwMatrix.reshape(self.minSize_*self.nDomains_,self.minSize_*self.nDomains_).swapaxes(0,1).reshape(self.numBlocks_[0],self.numBlocks_[1] , self.minSize_ )

        dwMatrix = cuda.to_device(hwMatrix)
        dPlist = cuda.to_device(hPlist)
        dQlist = cuda.to_device(hQlist)
        dRlist = hRlist
        
        dvjInvList = cuda.to_device(hvjInvList)
        ddjlist    = cuda.to_device(hdjlist)

        dquadCoeffsMatrix = cuda.to_device(hquadCoeffsMatrix)
        dnewUMatrix       = cuda.to_device(empty3dMatrix)
        
        uLocal2D[(self.numBlocks_[1],self.numBlocks_[0]),self.numThreads_](dwMatrix, dPlist 
             , dQlist            , dRlist
             , dvjInvList        , ddjlist 
             , dquadCoeffsMatrix , dJLvalMatrix 
             , dJRvalMatrix      , dnewUMatrix 
             , dleftComm         , drightComm 
             , dJSizeList)
        
        hyvals_ = emptyMatrix#numpy.zeros_like(hUvalsMatrix,np.dtype=np.float32)  
        ha_     = emptyMatrix
        hb_     = emptyMatrix
        
        for subDomain in range(self.nDomains_):
            yindex1 = subDomain*self.nDomains_
            yindex2 = (subDomain+1)*self.nDomains_
            
            #x grids are the same for every y currently
            segment = self.rectSegments_[(subDomain,0)]
            hyvals_[:,yindex1:yindex2] = segment.yvals_

            ha_[:,yindex1:yindex2]     = segment.ya_
            hb_[:,yindex1:yindex2]     = segment.yb_
 
        self.hyvals_= hyvals_

        self.hleftComm  = dleftComm.copy_to_host()
        self.hrightComm = drightComm.copy_to_host()
        
        self.hA_ = emptyMatrix
        self.hB_ = emptyMatrix
        
        #try to cudaIze this, in 1d it doesn't make sense but in 2d it does 
        for i in range(self.numBlocks[0]):
            self.updateTransmissionCoeffs(i,self.leftComm[i,:],self.rightComm[i,:])

        dA_ = cuda.to_device(self.hA_)
        dB_ = cuda.to_device(self.hB_)

        da_          = cuda.to_device(ha_)
        db_          = cuda.to_device(hb_)
        dxvals_      = cuda.to_device(hxvals_)
        dUPrevMatrix = cuda.to_device (hUPrevMatrix)
        
        uUpdate2D[self.numBLocks_,self.numThreads_](dnewUMatrix, dyvals_ , dA_,dB_,self.alpha_ , da_, db_)        

        hnewUMatrix = dnewUMatrix.copy_to_host()
        hnewUMatrix = hnewUMatrix.reshape(self.minSize_*self.nDomains_,self.minSize_*self.nDomains_).swapaxes(0,1).reshape(self.numBlocks_[0],self.numBlocks_[1] , self.minSize_ )
        
        dnewUMatrix = cuda.to_device(hnewUMatrix)
        dFinalNewU  = cuda.to_device(empty3dMatrix)

        uFinalize(dFinalNewU,dnewUMatrix,self.dt_,self.time_, dUPrevMatrix , self.beta_)

        for index in range(self.nDomains_):
            self.lineSegs_[index].updateSoln(hUValsMatrix[index],hUPrevMatrix[index])        
        
        print("done!")
    
    """
    after calculating the local contributions in cuda, we need to update
    the boundaries 
    """
    def updateTransmissionCoeffs(self,i,hleftComm,hrightComm,axis):
        
        JL = hrightComm[0]
        JR = hleftComm[0]
        
        distanceScalar = 1
                 
        for xdomain in range(1,self.nDomains_):
            #which domain are we in,since we have the y
            ydomain = math.floor(i/self.minSize_)
            segment  = self.rectSegs_[(xdomain,ydomain)]
            expAlphaDistance = math.exp(-segment.getAlphaDistance('x'))
            
            self.hA_[i,xdomain] = JL
            self.hB_[i, self.nDomains_-xdomain-1]=JR
            
            distanceScalar*=expAlphaDistance
            
            JL = JL*distanceScalar + hrightComm[segmentIndex]
            JR = JR*distanceScalar + hleftComm[segmentIndex]
        
        self.applyBCs(JL,JR)
        
        distanceScalar = 1
        
        for xdomain in range(1,self.nDomains_):
            #which domain are we in,since we have the y
            ydomain = math.floor(i/self.minSize_)
            segment  = self.rectSegs_[(xdomain,ydomain)]

            expAlphaDistance = math.exp(-segment.getAlphaDistance('x'))
            self.hA_[i,segmentIndex] += distanceScalar*self.BC1_
            self.hB_[i,self.nDomains_-segmentIndex-1] += distanceScalar*self.BC2_

            expAlphaDistance = math.exp(-segment.getAlphaDistance('x'))
            distanceScalar *=expAlphaDistance