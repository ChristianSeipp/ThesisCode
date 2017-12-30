# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:50:50 2017

@author: Chris
"""
import sys
sys.path.insert(0, 'C:/Users/Chris/Documents/MOLTPython/Final1Dsolver')

from WaveSolver import *
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from lineSolve import lineSolve

import time

def createDomains(a,b,nDomains, nTot):
    pointsPerDomain = nTot/nDomains
    stepSize = (b-a)/nDomains
    newDomains = [np.linspace(a+(i)*stepSize,a+(i+1)*stepSize,pointsPerDomain,dtype = np.float32) for i in range(nDomains)]

    return(newDomains)


T = .01*1000
a = -2
b = 2
c   = 1
nDomains = 1

dt = .01
f = lambda x: 0#numpy.exp(-72*((x-(a+b)/2.)/(b-a))**2.)
g = lambda x: 0#144/(b-a)**2*(x-(a+b)/2)*numpy.exp(  -72.*( (x-(a+b)/2) / (b-a) )**2)
N = 2033+8
xvals = np.linspace(a,b,N)
xvals2 = np.linspace(a,b,2048)

def CPUSolve(x,f,g,T,dt,c):
    solver =  waveHandler(x , f,g,T,dt,c,'periodic')
    uSoln = []
    a = time.time()
    for i in range( 1000 ):    
        solver.propogate()
    b = time.time()

    print( b - a , "timeCPU" )
    uSoln = solver.getSolns();
    return uSoln

def GPUSolve(x,f,g,c,dt,bcs = 'periodic'):
    solver = lineSolve(x, f, g , c , dt , bcs)
    uSoln = []
    a = time.time()
    for i in range( 1000 ):    
        solver.propogate()
    b = time.time()
    
    print( b - a , "timeGPU" )
    uSoln = solver.getSolns()
    #x = solver.getXSize()2
    return uSoln
    
def animate(i):
    line1.set_data(xvals,USolnCPU[i])
    line2.set_data(xvals,USolnGPU[i])
    return line1,line2

def init():
    line1.set_data([], [])
    line2.set_data([],[])
    return line1,line2

def main():
    USolnCPU = CPUSolve(xCpu,f,g,T,dt,c)
    #USolnCPU2 = CPUSolve(x2,f,g,T,dt,c)
    USolnGPU = GPUSolve(x,f,g,c,dt)
    #print(len(USolnCPU2),len(xvals))
    #solDiffs =    [ abs(USolnCPU[ i ] - USolnGPU[ i ]) for i in range( len( USolnGPU ) ) ]
    #print( USolnCPU , USolnGPU )
    #print( max( solDiffs ) , max( solDiffs) / USolnCPU[ solDiffs.index( max( solDiffs ) ) ]  , '{0:.16f}'.format(USolnCPU[ solDiffs.index( max( solDiffs ) ) ] ) , '{0:.16f}'.format( USolnGPU[ solDiffs.index( max( solDiffs ) ) ] ) ,  "relErr ")

nTot   = [  2**5 , 2**7 , 2**8 , 2**9 , 2**10 , 2**11 ,2**12  ]
for i in range( len( nTot ) ):
    
    x = createDomains(a,b,nDomains, nTot[ i ] )
    xCpu = createDomains(a , b ,  1 , nTot[ i ] )
    #x1 = np.linspace(-2, 0, N,dtype = np.float32)
    #x2 = n
    p = 144*2
    f = lambda x:  math.exp(-p*((x-(a+b)/2)/(b-a))**2)
    g = lambda x: 2*p/(b-a)**2*(x-(a+b)/2)*math.exp(  -p*( (x-(a+b)/2) / (b-a) )**2)
    #f = lambda x: math.cos(40*x)/(1+10*x**2)
    #g  = lambda x: 0
    c = 1
    
    main();
    
#uSolnDiff = [(np.abs(np.subtract(USolnCPU[i],USolnGPU[i]))) for i in range(1000) ]
#print(np.sum(uSoln))
#fig1 = plt.figure()
#fig1.plot(uSolnDiff)
#
#fig   = plt.figure()

#line   = plt.plot(.01*np.array(range(1000)),uSolnDiff)
#plt.show()
#ax    = plt.axes(xlim=(-2,2),ylim = (-1.3,4))
#line1, = ax.plot([],[] , lw=4,label='cpu 1 domain')
#line2, = ax.plot([],[], lw=2,color='r', label = 'gpu 16 domain')
#line3, = ax.plot([],[], lw=2 , color='g',label = 'cpu 16 domain')
#legend = plt.legend(loc='upper right')

#anim = animation.FuncAnimation(fig,animate, blit=True,frames = len(uSolnDiff),interval = 20)