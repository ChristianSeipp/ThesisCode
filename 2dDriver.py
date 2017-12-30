# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 21:55:03 2017

@author: Chris
"""

from WaveSolver2D import *

def createDomains(a,b,nDomains, nTot):
    pointsPerDomain = nTot/nDomains
    stepSize = (b-a)/nDomains
    newDomains = [np.linspace(a+(i)*stepSize,a+(i+1)*stepSize,pointsPerDomain,dtype = np.float32) for i in range(nDomains)]

    return(newDomains)
    
xa = -2
xb = 2

ya = -2
yb = 2

nxDomains = 1
nyDomains = 1

nTot = 2**8

c = 1

dt = .01

xDomains = createDomains(xa,xb,nxDomains,nTot)
yDomains = createDomains(ya,yb,nyDomains,nTot)

e = 72
f = lambda x , y :  math.exp(-e*((x-.5)**2+(y-.5)**2));
g = lambda x , y : 2*e*(x-y)*math.exp(-e*((x-.5)**2+(y-.5)**2))

bcs = 'periodic'

solver = WaveSolver2D(xDomains,yDomains,f,g,c,dt,bcs)
solver.timeStep()

