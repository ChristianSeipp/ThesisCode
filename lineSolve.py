from lineSeg import lineSeg
from CudaKernels import *
import math

"""
solve the wave equation on a 1D line

xdomains - a list of list of xdomains
f        - the function describing the initial state
g        - the function describing the initial time derivative
c        - speed of wave propogation
cfl      - relationship between the time and spatial discretization
bcs      - what type of boundary conditions do we have?
"""
class lineSolve:
    def __init__(self,xdomains, f, g , c , dt , bcs):

        self.beta_ = 2
        self.bcType_ = bcs
        self.lineSegs_ = []
        self.domainSize_ = xdomains[-1][-1]-xdomains[0][0]
        
        self.dt_ = dt
        #find out what the smallest x grid is
        self.minSize_ = len(xdomains[0])
        
        for xdomain in xdomains:
            dsize = len(xdomain)
            
            if dsize<self.minSize_:
                self.minSize_ = dsize
        
        #NOTE current behavior is to split and resize domains(upwards) until 
        #we have equal sized domains 

        for xdomain in xdomains:
            dsize = len(xdomain)

            if dsize != self.minSize_:
                newDomains = self.splitDomain(xdomain,dsize)
                
                for newDomain in newDomains:
                    newLineSeg = lineSeg(newDomain,f,g,c,dt)
                    
                    
                    self.lineSegs_.append(newLineSeg)
            else:
                newLineSeg = lineSeg(xdomain,f,g,c,dt)
                
                
                self.lineSegs_.append(newLineSeg)

        # these parameters are used in calculating the values at the boundary
        self.alpha_ = self.beta_/(c*self.dt_)
        self.damping_ = math.exp(-self.alpha_*(self.domainSize_))
        
        self.nDomains_ = len(self.lineSegs_)
        self.time_ = self.dt_
        
        print(self.nDomains_*self.minSize_)
        
        self.numThreads_ = self.nDomains_
        self.numBlocks_ = 1

        
        #figure out how to put it onto multiple blocks if it gets too big. 
        #shouldn't hit performance too much since there isn't much shared memory between
        #subdomains
        
        #one idea is to just make a 'struct' that holds the cuda device memory for each block,
        #then merge everything back together in the end
        
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
    """
    waveHandler should know about the BC's, segments shouldn't have any idea
    what the boundary conditions are, this will set the correct bc conditions
    """
    def applyBCs(self, leftEnds , rightEnds):
        if self.bcType_ == 'periodic':
            self.BC1_ = leftEnds/(1-self.damping_)
            self.BC2_ = rightEnds/(1-self.damping_)
    

    """
    need to put variables on the GPU, this should only get called once, use updateCUDAdeviceMemory
    for all other needs

    BECAUSE NUMBA has garbage collection we cannot manage memory as much as i'd like
    and have to do everything locally in this context
    """
    def calcTimeStep(self):
        self.time_ += self.dt_
        
        emptyMatrix    = np.zeros((self.nDomains_ , self.minSize_ ),dtype = np.float32)
        emptyList      = np.zeros(self.nDomains_,dtype=np.float32)


        ##NOTE All of the following values were written for convenience, 
        #      if it's a performance issue, put everything into 1 loop
        hxvals_ = np.array([segment.xvals_ for segment in self.lineSegs_],dtype =np.float32)
        self.hxvals_= hxvals_
        ha_     = np.array([segment.a_ for segment in self.lineSegs_], dtype = np.float32)
        hb_     = np.array([segment.b_ for segment in self.lineSegs_], dtype = np.float32)

        #store the information on the host
        hUValsMatrix  = np.array([segment.getState() for segment in self.lineSegs_], dtype = np.float32)
        hUPrevMatrix  = np.array([segment.uprev_ for segment in self.lineSegs_],dtype = np.float32)


        hPlist = np.array([segment.P_ for segment in self.lineSegs_],dtype = np.float32)
        hQlist = np.array([segment.Q_ for segment in self.lineSegs_],dtype = np.float32)
        hRlist = np.array([segment.R_ for segment in self.lineSegs_],dtype = np.float32)

        hvjInvList = np.array([1/segment.vj_**2 for segment in self.lineSegs_],dtype = np.float32)
        hdjlist    = np.array([segment.dj_ for segment in self.lineSegs_],dtype = np.float32)

        hquadCoeffsMatrix = emptyMatrix
        hJLvalMatrix = emptyMatrix
        hJRvalMatrix = emptyMatrix

        hwMatrix    = emptyMatrix
        hleftComm    = emptyList
        hrightComm  = emptyList

        hJSizeList  = np.array([len(segment) for segment in self.lineSegs_],dtype = np.uint32)#should be constant currently
        
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

        dJSizeList  = cuda.to_device(hJSizeList)
     
        uLocal1D[self.nDomains_,hJSizeList[0]](dUValsMatrix, dPlist 
             , dQlist            , dRlist
             , dvjInvList        , ddjlist 
             , dquadCoeffsMatrix , dJLvalMatrix 
             , dJRvalMatrix      , dwMatrix 
             , dleftComm         , drightComm 
             , dJSizeList)
        self.hleftComm  = dleftComm.copy_to_host()
        self.hrightComm = drightComm.copy_to_host()

        self.updateTransmissionCoeffs()

        dA_ = cuda.to_device(self.hA_)
        dB_ = cuda.to_device(self.hB_)

        da_          = cuda.to_device(ha_)
        db_          = cuda.to_device(hb_)
        dxvals_      = cuda.to_device(hxvals_)
        dUPrevMatrix = cuda.to_device (hUPrevMatrix)
           
        uUpdate1D[self.nDomains_, hJSizeList[0]](dUValsMatrix, dwMatrix, dxvals_ , self.dt_,
                   self.time_ , dUPrevMatrix,dA_,dB_,self.alpha_,self.beta_,da_,db_)

        hUValsMatrix = dUValsMatrix.copy_to_host()
        hUPrevMatrix  = dUPrevMatrix.copy_to_host()
        
        for index in range(self.nDomains_):
            self.lineSegs_[index].updateSoln(hUValsMatrix[index],hUPrevMatrix[index])
      
    
    """
    after calculating the local contributions in cuda, we need to update
    the boundaries 
    """
    def updateTransmissionCoeffs(self):
        self.hA_    = np.zeros_like(self.hleftComm,dtype = np.float32)
        self.hB_    = np.zeros_like(self.hrightComm,dtype = np.float32)
        
        JL = self.hrightComm[0]
        JR = self.hleftComm[0]
        
        distanceScalar = 1
        
        for segmentIndex in range(1,self.nDomains_):
            segment  = self.lineSegs_[segmentIndex]
            expAlphaDistance = math.exp(-segment.getAlphaDistance())
            
            self.hA_[segmentIndex] = JL
            self.hB_[self.nDomains_-segmentIndex-1]=JR
            
            distanceScalar*=expAlphaDistance
            
            JL = JL*distanceScalar + self.hrightComm[segmentIndex]
            JR = JR*distanceScalar + self.hleftComm[segmentIndex]
        
        self.applyBCs(JL,JR)
        
        distanceScalar = 1
        
        for segmentIndex , segment in enumerate(self.lineSegs_):
            self.hA_[segmentIndex] += distanceScalar*self.BC1_
            self.hB_[self.nDomains_-segmentIndex-1] += distanceScalar*self.BC2_

            expAlphaDistance = math.exp(-segment.getAlphaDistance())
            distanceScalar *=expAlphaDistance
    
    def propogate(self):
        self.calcTimeStep()
        self.meshSolns()
        
    def meshSolns(self):
        self.allSolns_ = []
        #to properly mesh grids together
        #if it's even, then take the left endpoint off 
        #if it's odd, then take the right endpoint off
        for segmentIndex,segment in enumerate(self.lineSegs_):
            if segmentIndex == 0:
                self.allSolns_+= list(segment.getState())
            else:
                self.allSolns_+=list(segment.getState()[1:len(segment)])

    def getXSize(self):
        self.allX_ = []
        #to properly mesh grids together
        #if it's even, then take the left endpoint off 
        #if it's odd, then take the right endpoint off
        for segmentIndex,segment in enumerate(self.hxvals_):
            if segmentIndex == 0:
                self.allX_+= list(segment)
            else:
                self.allX_+=list(segment[1:len(segment)])
        return self.allX_
    def getSolns(self):
        return self.allSolns_