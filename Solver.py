# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:55:45 2016

@author: Chris
"""

from  GF_QUAD import genExpIntegrate
import math
import numpy

c = 1
f = lambda x: math.exp(-( 2*x-1)**2*40)
g = lambda x: 0
U = lambda x,t: 1./2*( f(x-c*t)+f(x+c*t) - (f(x-c*t+1)+f(x+c*t-1)) + (f(x-c*t+2)+f(x+c*t-2))  )

a  = 1
N = 100
T  = 4;


tGrid = numpy.linspace(0,4,N)
xGrid = numpy.linspace(-a,a,N)

beta = 2
alpha = beta/(c*(tGrid[2] - tGrid[1] ) );

    
#no idea what's happening here
vL = [math.exp(-alpha*a+x) for x in xGrid]
vR = [math.exp(-alpha*a-x) for x in xGrid]

dN = math.exp(-alpha*2*a)
denom = 1/((1-dN)**2)
wL = [(vL[i] -dN*vR[i])*denom for i in range(len(vL))]
wR = [(vR[i] -dN*vL[i])*denom for i in range(len(vL))]

InitVals = [f(x) for x in xGrid]
prevVals = [0]*len(initVals)

for t in tGrid:
    I = genExpIntegrate(f,xGrid,alpha)
    I = [I[i]- I[0]*wL[i]-I[-1]*wR[i] for i in range(len(I)) ]
    u = [2*]
        
    #we have dirchlet boundary Conditions
    #put that shit in 
    I = [i for i in I]

