# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:56:54 2016

@author: Chris
"""
import math
import numpy
import matplotlib.pyplot as plt
import time

def uniform_gf_quad(fvals, xvals , alpha):

    JLval = [0]     
    JRval = [0]
    
    #assume uniform step size
    h = xvals[1] - xvals[0]
    
    size = len(xvals)    
    
    vj    = alpha*h
    dj    = math.exp(-vj) 
    ratio = (1-dj)/vj
    P     = 1 - ratio
    Q     = -dj + ratio
    R     = 1-dj-vj/2*(1+dj )
    
    
    quadCoeffs = []  
    for i in range(size):
        if i != 0 and i != (size-1):
            quadCoeffs.append((fvals[i-1] -2*fvals[i] + fvals[i+1])*1/vj**2 )
        elif i == 0:
            quadCoeffs.append((2*fvals[0] -5*fvals[1] +4*fvals[2] - fvals[3])*1/vj**2 )
        else:
            quadCoeffs.append((2*fvals[-1] -5*fvals[-2] +4*fvals[-3] - fvals[-4])*1/vj**2)
        #evaluate the polynomial integral for each J between 0 and N.
    for j in range(size):
        
        if j != 0:
            #recursive formula to update the value of J
            JLcurr  = P*fvals[j] + Q*fvals[j-1] + quadCoeffs[j]*R
            JLval.append( dj*JLval[j-1] + JLcurr )
        if j != size-1:
            JRreverse = P*fvals[-(j+2)]+Q*fvals[ -(j+1) ] + R*quadCoeffs[-(j+1)] 
            JRval.append(dj*JRval[j] + JRreverse)

    JRval.reverse()
    #
    return [JLval[i]+JRval[i]  for i in range(size)]

T = 4
N = 101
a = -2
b = 2

x = numpy.linspace(a,b,N)

CFL = 1
c = 1

dx = x[1]-x[0]
dt = dx/c*CFL

beta = 2
alpha = beta /(c*dt)

f = lambda x: math.exp(-72*((x-(a+b)/2)/(b-a))**2)
g = lambda x: 144/(b-a)**2*(x-(a+b)/2)*math.exp(-72*((x-(a+b)/2)/(b-a))**2)

u0 = [f(xi) for xi in x ]
u1 = [f(xi)for xi in x]

t0 = 0

Ga = [math.exp(-alpha*(xi-a)) for xi in x]
Gb = [math.exp(-alpha*(b-xi)) for xi in x]
dN = [math.exp(-alpha*(b-a)) for xi in x]

vL = [(Ga[i]-dN[i]*Gb[i])/(1-dN[i]**2) for i in range(len(Ga))]
vR = [(Gb[i]-dN[i]*Ga[i])/(1-dN[i]**2) for i in range(len(Ga))]
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x,u1,'b-')
while t0<T:
    
    w = uniform_gf_quad(u1,x,alpha)
    wsize = len(w)
#    w = [w[i]- w[0]*vL[i]-w[-1]*vR[i] for i in range(wsize)]
    w = [u1[i] -.5*w[i] for i in range(wsize)]
    u = [2*u1[i] - u0[i]-beta**2*w[i] for i in range(wsize)]
    u0 = u1
    u1 = u    
    t0+=dt

#genExpIntegrate(f,x,alpha)