# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 04:21:25 2016

@author: Chris
"""
import numpy as np
import math
"""
This is an object which will solve a segment of a distributed wave soln

xvals - the domain which is covered(Note: only uniform grids are allowed)
f     - the function describing the initial state
g     - the function describing the initial time derivative
alpha - decay parameter
c     - speed of propogation c 
cfl   - cfl condition
"""
class lineSeg:
    def __init__(self, xdomain, f,g,c,dt):
        self.beta_  = 2
        self.xvals_ = xdomain
        self.h_    = self.xvals_[1]-self.xvals_[0]
        self.a_     = self.xvals_[0]
        self.b_     = self.xvals_[-1]
        #self.dt_    = self.h_/c*cfl
        #self.alpha_ = self.beta_/(c*self.dt_)
        self.uprev_ = np.zeros_like(self.xvals_,dtype = np.float32)
        self.ucurr_ = np.zeros_like(self.xvals_,dtype = np.float32)
        self.c_     = c
        self.f_     = f
        self.g_     = g
        self.dt_    = dt
        
        self.setCFL()
        self.calcUniformQuadWeights()
        self.taylorStep( )
       
    def getAlphaDistance(self):
        return self.alpha_*(self.b_-self.a_)
    
    def __len__(self):
        return len(self.xvals_)
    """the first timestep is done with a 2nd order taylor expansion"""
    def taylorStep(self):
        #we need to set the initial state 
        for i , xi in enumerate(self.xvals_):
            fval = np.float32(self.f_(xi))
            self.uprev_[i] = fval
            #self.ucurr_.append(fval + self.dt_*g(xi))
            self.ucurr_[i] = np.float32(1/2*(self.f_(xi-self.c_*self.dt_)+ self.f_(xi+self.c_*self.dt_) + self.dt_*(self.g_(xi-self.c_*self.dt_)+self.g_(xi+self.c_*self.dt_))))
        #self.ucurr_[0] = (self.ucurr_[0]+self.ucurr_[-1])/2
        #self.ucurr_[-1] = self.ucurr_[0]
    
    """time steps cannot be determined until all subgrids are made"""
    def setCFL(self):
        self.CFL = self.dt_*self.c_/self.h_
        print (self.CFL)
        self.alpha_ = self.beta_/(self.c_*self.dt_) 
        

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
    
    def getDist(self):
        return self.h_
    
    def updateSoln(self, newU,newPrev):
        self.ucurr_ = newU
        self.uprev_ = newPrev