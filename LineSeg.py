# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:48:08 2017

@author: Chris
"""

import math
import numpy
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from numba import cuda
#number of threads per block
NTHREADS = 128

"""
This is an object which will solve a segment of a distributed wave soln

xvals  the domain which is covered(Note: only uniform grids are allowed)
f      the function describing the initial state
g      the function describing the initial time derivative
alpha  decay parameter
c      speed of propogation c 
cfl   cfl condition
"""
class lineSegment:
    """set the state variables and step forward in time once  """
    def __init__(self , xdomain, f , g , c, dt ):
        self.dt_    = dt
        self.beta_  = 2
        self.xvals_ = xdomain
        self.h_    = self.xvals_[1]-self.xvals_[0]
        self.a_     = self.xvals_[0]
        self.b_     = self.xvals_[-1]
        #self.dt_    = self.h_/c*cfl
        self.alpha_ = self.beta_/(c*self.dt_)
        self.uprev_ = []
        self.ucurr_ = []
        self.c_     = c
        self.f_     = f
        self.g_     = g
        
        self.setCFL()
        self.dt_    = dt
        #self.s_ = lambda x: 10*(math.tanh(20*x-40)**2-math.tanh(20*x-20)**2)
        self.s_ = lambda x: 0
    def getAlphaDistance(self):
        return self.alpha_*(self.b_-self.a_)
    
    """time steps cannot be determined until all subgrids are made"""
    def setCFL(self):
        self.CFL = self.dt_*self.c_/self.h_
    
    def __len__(self):
        return len(self.xvals_)
        
    """the first timestep is done with a 2nd order taylor expansion"""
    def taylorStep(self):
        #we need to set the initial state 
        for xi in self.xvals_:
            fval = self.f_(xi)
            self.uprev_.append(fval)
            #self.ucurr_.append(fval + self.dt_*g(xi))
            self.ucurr_.append(1/2*(self.f_(xi-self.c_*self.dt_)+ self.f_(xi+self.c_*self.dt_) + self.dt_*(self.g_(xi-self.c_*self.dt_)+self.g_(xi+self.c_*self.dt_))))
        #self.ucurr_[0] = (self.ucurr_[0]+self.ucurr_[-1])/2
        #self.ucurr_[-1] = self.ucurr_[0]
    
    """time steps cannot be determined until all subgrids are made"""
    def setTimestep(self , dt):
        self.dt_ = dt
        self.alpha_ = self.beta_/(self.c_*self.dt_) 
        
        self.calcUniformQuadWeights()
        self.taylorStep()
       
    """the the mesh is not adaptive, many constants are the same"""
    def calcUniformQuadWeights( self ):
            
            #assume uniform step size
            
            self.vj_    = self.alpha_*self.h_
            self.dj_    = math.exp(-self.vj_) 
            
            ratio = (1-self.dj_)/self.vj_
            eps = 1e-3
            self.Q_     = -self.dj_ + ratio

            #if things get small, this blows up 
            if self.vj_>eps:
                self.P_ = 1 - ratio
                self.R_ = 1-self.dj_-self.vj_/2*(1+self.dj_ )
            else:
                self.P_ = (self.h_/2-self.h_**2/6+self.h_**3/24-self.h_**4/120)
                self.R_ = -math.exp(-self.h_/2)*(self.h_**3/12+self.h_**5/480+self.h_**7/53760)

    """get the current value of u"""
    def getState(self):
        return self.ucurr_
    
    """
    calculate the local contribution of the solution and report the endpoints
    THIS WILL LIKELY BE THE THING TO PARALLELIZE
    """
    def calcLocalContribution( self, uvals = -1):
        if uvals == -1:
            uvals = self.ucurr_
        
        quadCoeffs = []  
        size = len(self.xvals_)    
        
        JLval = [0]
        JRval = [0]
        
        #do a finite difference stencil for the second derivative at every point
        #in our mesh. do a 4 point stencil at the ends to preserve accuracy
        for i in range(size):
            if i != 0 and i != (size-1):
                quadCoeffs.append((uvals[i-1] - 2*uvals[i] + uvals[i+1])*1/self.vj_**2 )
            elif i == 0:
                quadCoeffs.append((2*uvals[0] - 5*uvals[1] + 4*uvals[2] - uvals[3])*1/self.vj_**2 )
            else:
                quadCoeffs.append((2*uvals[-1] - 5*uvals[-2] +4*uvals[-3] - uvals[-4])*1/self.vj_**2)
            #evaluate the polynomial integral for each J between 0 and N.
        for j in range(size):

            if j != 0:
                #recursive formula to update the value of J
                JLcurr  = self.P_*uvals[j] + self.Q_*uvals[j-1] + quadCoeffs[j]*self.R_
                JLval.append( self.dj_*JLval[j-1] + JLcurr )
                
            if j != size-1:
                JRreverse = self.P_*uvals[-(j+2)]+self.Q_*uvals[ -(j+1) ] + self.R_*quadCoeffs[-(j+2)] 
                JRval.append(self.dj_*JRval[j] + JRreverse)
            
        JRval.reverse()
        
        self.w_ = [JLval[i]+JRval[i]  for i in range(size)]
        return self.reportEndPoints()
    
    """update the solution with the endpoints from it's neighbors(or boundaries)"""
    def updateSoln(self , Am ,Bm,tCurr):
        size = len(self.xvals_)
        self.u_ = [0]*size
        for i in range(size):
            x = self.xvals_[i]
            self.u_[i] = self.w_[i]+Am*math.exp(-self.alpha_ *(x -self.a_ ) ) + Bm*math.exp(-self.alpha_ *(self.b_  - x))
            self.u_[i] += self.s_(tCurr)*self.dt_*math.exp(-self.alpha_*abs(x))
    """after ALL other calculations are done,  """
    
    def calcFinalSoln(self):
        size = len(self.xvals_)
        
        for i in range(size):        
            self.u_[i] = self.ucurr_[i] -.5*self.u_[i]
            self.u_[i] = 2*self.ucurr_[i] - self.uprev_[i] - self.beta_**2*self.u_[i]
            
        self.uprev_ = self.ucurr_
        self.ucurr_ = self.u_
    
    """pass the end points of the segment up to the global solver"""
    def reportEndPoints(self):
        return [self.w_[0] , self.w_[-1]]
    
    """report the solution at a given timestep"""
    def getSoln(self):
        return self.ucurr_
    
    def getDist(self):
        return self.h_