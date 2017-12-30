# -*- coding: utf-8 -*-

from LineSeg import *
import math
"""
Created on Fri Jan  6 16:49:54 2017

@author: Chris
"""

"""
class to solve wave equation in a distributed manner
xdomain is a list of lists that contains x vals, each sublist must be monotonicly increasing
f       is the initial position of the wave at time 0
g       is the initial velocity of the wave at time 0
T       is the end time
cfl     is a relationship between the time and spatial discretization
c       is the speed of wave propogation
bcs     is the desired boundary conditions, options are "neu", "dirch", and "period"
"""
#NOTE: may want to change xdomains to [[a1,b1],[a2,b2]...] for more convenience
class waveHandler:
    def __init__( self , xdomains, f ,g, T, dt , c , bcs):
        
        self.beta_ = 2
        self.segments   = []
        self.dt_        = dt
        self.end_       = T
        
        self.leftComm_   = []
        self.rightComm_  = []
        self.Ileft_      = [0]
        self.Iright_     = [0]
        self.domainSize_ = xdomains[-1][-1]-xdomains[0][0]
        
        self.tCurr_ = 2*dt
        self.bcType_ = bcs
        
        #create a segment for each seperate domain
        for xdomain in xdomains:
            newSegment = lineSegment(xdomain , f , g , c, dt )
            self.segments.append(newSegment)
       
        self.alpha_ = self.beta_/(c*self.dt_)
        self.damping_ = math.exp(-self.alpha_*(self.domainSize_))

        for currsegment in self.segments:
            currsegment.setTimestep(self.dt_)
        
        self.segmentLen = len(self.segments)
        self.time_ = 2*self.dt_
        #need to figure out how to make time steps should it be based off of largest things?
        
    """
    propogate the solution to the specified time. Maybe have another option
    for arbitrary time stepping?
    """
    def propogate(self):
        self.calcTimeStep()
        self.meshSolns()
        self.tCurr_ += self.dt_
            
    """
    waveHandler should know about the BC's, segments shouldn't have any idea
    what the boundary conditions are, this will set the correct bc conditions
    """
    def applyBCs(self, leftEnds , rightEnds):
        if self.bcType_ == 'periodic':
            self.BC1_ = leftEnds/(1-self.damping_)
            self.BC2_ = rightEnds/(1-self.damping_)
    
    """do local contributions for each segment, and then calculate """
    def calcTimeStep(self):
        #need to reset the 'memory'
        self.leftComm_  = [0]*self.segmentLen
        self.rightComm_ = [0]*self.segmentLen
        
        self.A_    = [0]*self.segmentLen
        self.B_    = [0]*self.segmentLen
        #BCends = [ 0 , 0]
        
        #calculate local contributions, and report the end points
        for segmentIndex in range(self.segmentLen):
            segment   = self.segments[segmentIndex]
            
            endpoints = segment.calcLocalContribution()

            self.leftComm_[self.segmentLen-segmentIndex-1] = endpoints[0]
            self.rightComm_[segmentIndex] = endpoints[1]

        JL = self.rightComm_[0]
        JR = self.leftComm_[0]
        #print(self.rightComm_,self.leftComm_)
        
        distanceScalar = 1
        
        for segmentIndex in range(1,self.segmentLen):
            segment  = self.segments[segmentIndex]
            expAlphaDistance = math.exp(-segment.getAlphaDistance())
            
            self.A_[segmentIndex] = JL
            self.B_[self.segmentLen-segmentIndex-1]=JR
            
            distanceScalar*=expAlphaDistance
            
            JL = JL*distanceScalar + self.rightComm_[segmentIndex]
            JR = JR*distanceScalar + self.leftComm_[segmentIndex]
        self.applyBCs(JL,JR)

        distanceScalar = 1
        for segmentIndex , segment in enumerate(self.segments):
            self.A_[segmentIndex] += distanceScalar*self.BC1_
            self.B_[self.segmentLen-segmentIndex-1] += distanceScalar*self.BC2_

            expAlphaDistance = math.exp(-segment.getAlphaDistance())
            distanceScalar *=expAlphaDistance
        
        #self.B_.reverse
        #print(self.A_,self.B_)
        for segmentIndex in range(self.segmentLen):
            currSegment  = self.segments[segmentIndex]
            currSegment.updateSoln(self.A_[segmentIndex], self.B_[segmentIndex],self.tCurr_)
            currSegment.calcFinalSoln()
    
    def meshSolns(self):
        self.allSolns_ = []
        #to properly mesh grids together
        #if it's even, then take the left endpoint off 
        #if it's odd, then take the right endpoint off
        for segmentIndex,segment in enumerate(self.segments):
            if segmentIndex == 0:
                self.allSolns_+= list(segment.getSoln())
            else:
                self.allSolns_+=list(segment.getSoln()[1:len(segment)])
        
    def getSolns(self):
        return self.allSolns_
        #segments[0].applyBCs()
        #segment[-1].applyBcs()