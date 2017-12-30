# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 04:19:18 2016

@author: Chris
"""
from numba import cuda
from numba import void, uint8 , uint32, uint64, int32, int64, float32, float64, f8
import numpy as np
import math

device = cuda.get_current_device()

@cuda.jit('void(float32[:],float32,float32,float32,float32,float32,float32[:],float32[:] ,float32[:], uint32)',device = True)
def calcLocalJs(uvals,P,Q,R,vjsqrinv,dj,quadCoeffs,JLvals,JRvals ,JSize):
        #do a finite difference stencil for the second derivative at every point
        #in our mesh. do a 4 point stencil at the ends to preserve accuracy
        j = cuda.threadIdx.x

        if j != 0 and j != (JSize-1):
            quadCoeffs[j]= (uvals[j-1] - 2*uvals[j] + uvals[j+1])*vjsqrinv 
        elif j == 0:
            quadCoeffs[j] = (2*uvals[0] - 5*uvals[1] + 4*uvals[2] - uvals[3])*vjsqrinv
        else:
            quadCoeffs[j] = (2*uvals[JSize-1] - 5*uvals[JSize-2] +4*uvals[JSize-3] - uvals[JSize-4])*vjsqrinv
        #evaluate the polynomial integral for each J between 0 and N.

        if j != 0:
            #recursive formula to update the value of J
            JLcurr  = P*uvals[j] + Q*uvals[j-1] + quadCoeffs[j]*R
            JLvals[j] = JLcurr

        if j != JSize-1:
            JRreverse = P*uvals[-(j+2)]+Q*uvals[ -(j+1) ] + R*quadCoeffs[-(j+2)] 
            JRvals[JSize-j-2] = JRreverse

"""
solve the local pieces of the wave equation in 1d and return the end points of this calculation
each thread is going to solve a local piece of the wave equation and return an end point

a list or matrix of values is passed in where each column of the matrix and list corresponds
to the value of that segment

after we are done computing the local contributions, pass the values back to the cpu to calculate
the nonlocal contribution values

INPUTS:
uvalsMatrix      - a 2d Array of the value of u for each segment
Plist            - the values of P in our quadrature
Qlist            - the values of Q in our quadrature
Rlist            - the values of R in our quadrature
vjInvList        - the values of 1/vj**2 in our quaderature
djlist           - the values of dj in our quadrature
quadCoeffsMatrix - an empty 2d array of size(uvalsMatrix) that stores finite difference stencils
JLValMatrix      - an empty 2d array of size(uvalsMatrix) that stores the lefthand convolution
JRValMatrix      - an empty 2d array of size(uvalsMatrix) that stores the righthand convolution
JSizeList        - how large is each uvals array (may be unneccessary should be constant currently)

OUTPUTS:
wMatrix   - a 2d array of size(uvalsMatrix) that stores the sum of left+right convolution
leftComm  - a 1d array of size(domainCount) that stores the left side communication coefficients
rightComm - a 1d array of size(domainCount) that stores the right side communication coefficients

"""
@cuda.jit('void(float32[:,:],float32[:],float32[:],float32[:],float32[:] , float32[:], float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:],float32[:],uint32[:])')
def uLocal1D(uvalsMatrix, Plist , Qlist , Rlist
             , vjInvList , djlist 
             , quadCoeffsMatrix , JLvalMatrix 
             , JRvalMatrix      , wMatrix 
             , leftComm          , rightComm 
             , JSizeList):
    
    i = cuda.blockIdx.x
    j = cuda.threadIdx.x
    JSize    = JSizeList[i]    
    P        = Plist[i]
    Q        = Qlist[i]
    R        = Rlist[i]
    vjsqrinv = vjInvList[i]
    dj       = djlist[i]
    
    calcLocalJs(uvalsMatrix[i,:],P,Q,R,vjsqrinv,dj,quadCoeffsMatrix[i,:],JLvalMatrix[i,:],JRvalMatrix[i,:] , JSize)
    
    cuda.syncthreads()
    #wait for all threads to finish
    
    #this seems like a bad way to do this,
    #if we are on the first thread 
    if j == 1:
        index =0
        
        while index < JSize:
            if index != 0:
                JLvalMatrix[i][index] = dj*JLvalMatrix[i][index-1]+JLvalMatrix[i][index]
            index += 1
    elif j == 2:
        index = 0
        while index < JSize:
            if index !=JSize-1:
                reverseIndex = JSize-index-1
                JRvalMatrix[i][reverseIndex-1] = dj*JRvalMatrix[i][reverseIndex] + JRvalMatrix[i][reverseIndex-1]
            index += 1
    
    cuda.syncthreads()
    
    wMatrix[i,j] = (JLvalMatrix[i,j] + JRvalMatrix[i,j])
    
    cuda.syncthreads()
    
    if j == 1:
        leftComm[len(uvalsMatrix) - i - 1] = wMatrix[i][0]
        rightComm[i]  = wMatrix[i][JSize-1]
        
    cuda.syncthreads()
    #we should have a different P,Q,R  , vj, and dj for each sub domain, 

"""
this function should change to whatever the source function is 
f(x,y)*dt*exp(-alpha*abs(x-midpoint))
"""
@cuda.jit('float32(float32,float32,float32,float32,float32)',device = True)
def sourceTerm(dt,time,alpha,x,wVal):
    return 10*(math.tanh(20*time-40)**2-math.tanh(20*time-20)**2)*dt*math.exp(-alpha*abs(x))
    #print (wVal)

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32,float32, float32[:,:],float32[:],float32[:],float32, float32,float32[:],float32[:])')
def uUpdate1D(uvalsMatrix, wMatrix, xvals ,dt, time, uPrevMatrix,A,B,alpha,beta,a,b):
    j = cuda.threadIdx.x
    i = cuda.blockIdx.x
    
    wMatrix[i,j] = wMatrix[i,j]+A[i]*math.exp(-alpha*(xvals[i,j] - a[i] ) ) + B[i]*math.exp(-alpha *(b[i]  - xvals[i,j]))
    
    wMatrix[i,j] += sourceTerm(dt,time,alpha,xvals[i,j] , wMatrix[i,j])
    
    wMatrix[i,j] = uvalsMatrix[i,j] - .5*wMatrix[i,j]
    wMatrix[i,j] = 2*uvalsMatrix[i,j] - uPrevMatrix[i,j] - beta**2*wMatrix[i,j]
    uPrevMatrix[i,j] = uvalsMatrix[i,j]
    uvalsMatrix[i,j] = wMatrix[i,j]
    
    cuda.syncthreads()

