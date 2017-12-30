# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 02:09:22 2016

@author: Chris
"""

from quadWeights import *
import numpy as np

"""
This class stores 2d segments of our wave solver, note that it doesn't solve anything, just stores
X , Y , U and a bunch of parameters that get passed to a 1d solver

xdomain x points
ydomain y points
f   function describing the initial state  (currently radial )
g   function describing the initial derivative in time  (currently radial)
c   speed of wave propogation
cfl time propogation length factor
dt  time step length

"""
class rectSegment:
    def __init__(self , xdomain,ydomain, f , g, c , dt):
        self.c_  = c
        self.f_   = f
        self.g_   = g
        
        self.beta_ = 2
        

        
        self.x_ = xdomain
        self.y_ = ydomain
        self.N_ = len(xdomain)
        
        
        self.xa = self.x_[0]
        self.xb = self.x_[-1]
        self.ya = self.y_[0]
        self.yb = self.y_[-1]
        
        self.xDist_ = self.x_[-1] - self.x_[0]
        self.yDist_ = self.y_[-1] - self.y_[0]
        
        self.dx_ = self.x_[1] - self.x_[0]
        self.dy_ = self.y_[1] - self.y_[0]
        self.dt_ = dt
        
        self.X_ , self.Y_ = np.meshgrid(self.x_ , self.y_)
        
        self.f_  = f
        self.g_  = g
        
        #w is the intermediate solution
        self.w_ = np.zeros((self.N_,self.N_))
        self.setCFL()

        
    """time steps cannot be determined until all subgrids are made"""
    def setCFL(self ):
        self.xCFL_ = self.dt_*self.c_/self.dx_
        self.yCFL_ = self.dt_*self.c_/self.dy_
        
        self.alpha_ = self.beta_/(self.c_*self.dt_) 
        self.alphaXDist_ = self.alpha_*self.xDist_
        self.alphaYDist_ = self.alpha_*self.yDist_
        
        self.xquadWeights_ = quadWeights(self.alpha_ , self.dx_ )
        self.yquadWeights_ = quadWeights(self.alpha_ , self.dy_ )

        self.taylorStep( )
       
    
    """step forward once in time using a different method (in this case a taylor series)"""
    def taylorStep( self ):
        N = self.N_
        self.ucurr_ = np.zeros((N,N))
        self.uprev_ = np.zeros((N,N))
        
        for  xindex, x in enumerate(self.x_):
            
            
            for yindex , y in enumerate(self.y_):
                self.uprev_[xindex,yindex] = self.f_( x , y )
                self.ucurr_[xindex,yindex] = self.f_(x , y) + self.dt_*self.g_( x , y )
    
    def storeIntermSoln( self , col , intermSoln):
        for rowIndex , row in enumerate(self.w_):
            row[col] = intermSoln[rowIndex]
    
    def updateSoln( self , row , newSoln):
        self.uprev_[row] = self.ucurr_[row]
        self.ucurr_[row] = newSoln
 
    def plotSoln( self , ax):
        surf = ax.plot_surface(self.X_, self.Y_ ,self.ucurr_,rstride=2, cstride=2, shade=False, cmap="jet", linewidth=1)
        surf.set_edgecolors("black")
        surf.set_facecolors(surf.to_rgba(surf._A))
    
    def getP(self,axis):
        if axis == 'x':
            quadWeights = self.xquadWeights_
        elif axis == 'y':
            quadWeights = self.yquadWeights_
        return quadWeights.P_
    
    def getQ(self,axis):
        if axis == 'x':
            quadWeights = self.xquadWeights_
        elif axis == 'y':
            quadWeights = self.yquadWeights_
        return quadWeights.Q_
    
    def getR(self,axis):
        if axis == 'x':
            quadWeights = self.xquadWeights_
        elif axis == 'y':
            quadWeights = self.yquadWeights_
        return quadWeights.R_
    
    def getVj(self,axis):
        if axis == 'x':
            quadWeights = self.xquadWeights_
        elif axis == 'y':
            quadWeights = self.yquadWeights_
        return quadWeights.vj_

    def getDj(self,axis):
        if axis == 'x':
            quadWeights = self.xquadWeights_
        elif axis == 'y':
            quadWeights = self.yquadWeights_
        return quadWeights.dj_

    def getAlphaDistance(self,axis):
        if axis == 'x':
            return self.alphaXDist_
        else:
            return self.alphaYDist_
        