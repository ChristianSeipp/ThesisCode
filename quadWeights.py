# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 01:57:34 2016

@author: Chris
"""
import math
import numpy as np
"""
quaderature weights stay the same between all x grids in a subdomain, and all y grids in a subdomain
having this prevents having alot of copy pasted code
#NOTE: go through and replace a bunch of stuff with numpy 2darrays instead of lists

"""
class quadWeights:
    def __init__(self, alpha , h ):
            self.alpha_ = np.float32(alpha)
            self.h_     = np.float32(h)
            
            self.vj_    = self.alpha_*self.h_
            self.dj_    = np.exp(-self.vj_) 
            
            ratio = (1-self.dj_)/self.vj_
            eps = 1e-3
            self.Q_     = np.float32(-self.dj_ + ratio)

            #if things get small, this blows up 
            if self.vj_>eps:
                self.P_ = np.float32(1 - ratio)
                self.R_ = np.float32(1-self.dj_-self.vj_/2*(1+self.dj_ ))
            else:
                self.P_ = np.float32(self.h_/2-self.h_**2/6+self.h_**3/24-self.h_**4/120)
                self.R_ = np.float32(-math.exp(-self.h_/2)*(self.h_**3/12+self.h_**5/480+self.h_**7/53760))

        