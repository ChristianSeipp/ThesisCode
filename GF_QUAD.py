# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:45:33 2016

@author: Chris
"""

import math
import numpy
from matplotlib import pyplot as plt




"""
f       function being integrated
sGrid   spatial grid of length N+1 ( splits up the domain [0 1]) distance between point
alpha   constant for exponent


Evaluates the integral \int_0^1 exp( -alpha *z )*f(z) dz using 
quadratic quadrature

"""
def expIntegrate(fvals, sGrid ,  alpha):
    
    JLval = [0]     
    JRval = [0]
    
    N = len(sGrid) - 1
    
    spacing = [ sGrid[i+1] - sGrid[i] for i in range(N) ]
    
    vj    = []
    dj    = []
    ratio = []    
    P     = []
    Q     = []
    R     = []    
    quadCoeffs = []
    
    #iterate through entire grid            
    for j in range(0,N+1):
        if j == 0:
            quadCoeff = 2*
            quadCoeff = 2*spacing[0]**2*fvals[2]/(spacing[1]*(spacing[0]+spacing[1])) 
            quadCoeff -= 2*spacing[0]/spacing[1]*fvals[1] 
            quadCoeff += 2*spacing[0]/(spacing[0]+spacing[1])*fvals[0]     
        elif j != N:            
            # coefficient in quadratic interpolation
            quadCoeff = 2*spacing[j]**2*fvals[j+1]/(spacing[j]*(spacing[j-1]+spacing[j])) 
            quadCoeff -= 2*spacing[j-1]/spacing[j]*fvals[j] 
            quadCoeff += 2*spacing[j-1]/(spacing[j-1]+spacing[j])*fvals[j-1]
        elif j == N:
            quadCoeff = 2*spacing[j-2]**2*fvals[j]/(spacing[j-1]*(spacing[j-2]+spacing[j-1])) 
            quadCoeff -= 2*spacing[j-2]/spacing[j-1]*fvals[j-1] 
            quadCoeff += 2*spacing[j-2]/(spacing[j-2]+spacing[j-1])*fvals[j-2]
            
        quadCoeffs.append(quadCoeff)
        
        # coefficients for evaluating integral of quadratic
        vj.append( alpha*spacing[j] ) 
        dj.append( math.exp(-vj[j]) )
        
        #this division is used multiple times, jut do it once
        ratio.append( (1-dj[j])/vj[j])
        P.append( 1 - ratio[j] )
        Q.append( -dj[j] + ratio[j])
        R.append(1-dj[j]-vj[j]/2*(1+dj[j] ))
        
    
    #evaluate the polynomial integral for each J between 0 and N.
    for j in range(0,N+1):
        
        if j != 0:
            #recursive formula to update the value of J
            JLcurr  = P[j-1]*fvals[j-1] +Q[j-1]*fvals[j] + quadCoeffs[j-1]*R[j-1]
            JLval.append( dj[j-1]*JLval[j-1] + JLcurr)
        if j != N:
            JRreverse = P[-(j) ]*fvals[-(j+1)]+Q[ -(j) ]*fvals[ -j ] + R[ -j ]*quadCoeffs[-(j+1)] 
            JRval.append(dj[-j]*JRval[j] + JRreverse)

    JRval.reverse()
    # 
    return [JLval[i] +JRval[i] for i in range(0,N+1)]

"""
f      function to integrate 
xgrid  must be monotonically increasing grid
alpha  parameter for integration

general integration from a to b of alpha*\int_a^b exp(-alpha*|x-y|)*f(y)dy
"""
def genExpIntegrate(f, xGrid, alpha):

    fvals = [f(x) for x in xGrid]
 
    I = expIntegrate(fvals,xGrid,alpha) 
   
    return I


#test it
#f = lambda x: 1;  
#n = numpy.linspace(0,1,51)
#I = genExpIntegrate(f,n,1)
#plt.plot(n , I)
#plt.show()