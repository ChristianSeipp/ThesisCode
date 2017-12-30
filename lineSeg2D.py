# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 02:10:56 2016

@author: Chris
"""

"""
stores a 1d slice of a 2d line
"""
class lineSeg2D:
    def __init__(self, xdomain, ucurr,uprev,quadWeights,alpha,dt):
        self.beta_  = 2

        self.xvals_ = xdomain

        self.h_    = self.xvals_[1]-self.xvals_[0]

        self.a_     = self.xvals_[0]
        self.b_     = self.xvals_[-1]

        self.uprev_ = uprev
        self.ucurr_ = ucurr
        
        self.quadWeights_ = quadWeights
        self.alpha_ = alpha
        
        self.dt_    = dt
       
    def getAlphaDistance(self):
        return self.alpha_*(self.b_-self.a_)
    
    def __len__(self):
        return len(self.xvals_)

    """get the current value of u"""
    def getState(self):
        return self.ucurr_
    
    def getDist(self):
        return self.h_
    
    def updateSoln(self, newU,newPrev):
        self.ucurr_ = newU
        self.uprev_ = newPrev