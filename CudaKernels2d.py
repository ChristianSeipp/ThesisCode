# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 04:04:22 2016

@author: Chris
"""
import math
from numba import cuda
from numba import void, uint8 , uint32, uint64, int32, int64, float32, float64, f8
device = cuda.get_current_device()

@cuda.jit#('void(float32[:],float32,float32,float32,float32,float32,float32[:],float32[:] ,float32[:], uint32)',device = True)
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
Qlist            - the values of Q * our quadrature
Rlist            - the values of R in our quadrature
vjInvList        - the values of 1/vj**2 in our quaderature
djlist           - the values of dj in our quadrature
quadCoeffsMatrix - an empty 2d array of size(uvalsMatrix) that stores finite difference stencils
JLValMatrix      - an empty 2d array of size(uvalsMatrix) that stores the lefthand convolution
JRValMatrix      - an empty 2d array of size(uvalsMatrix) that stores the righthand convolution
JSize            - how large is each uvals array (should be constant currently)

OUTPUTS:
wMatrix   - a 2d array of size(uvalsMatrix) that stores the sum of left+right convolution
leftComm  - a 1d array of size(domainCount) that stores the left side communication coefficients
rightComm - a 1d array of size(domainCount) that stores the right side communication coefficients

"""
@cuda.jit#('void(float32[:,:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:] , float32[:,:], float32[:,:,:],float32[:,:,:],float32[:,:,:],float32[:,:,:],float32[:,:],float32[:,:],uint32)')
def uLocal2D(uvalsMatrix, Plist , Qlist , Rlist
             , vjInvList , djlist 
             , quadCoeffsMatrix , JLvalMatrix 
             , JRvalMatrix      , wMatrix 
             , leftComm          , rightComm 
             , JSize):
    JSize = 64
    j = cuda.blockIdx.x
    i = cuda.blockIdx.y

    k = cuda.threadIdx.x  
    
    P        = Plist[i,j]
    Q        = Qlist[i,j]
    R        = Rlist[i,j]
    vjsqrinv = vjInvList[i,j]
    dj       = djlist[i,j]
    
    calcLocalJs(uvalsMatrix[i,j,:],P,Q,R,vjsqrinv,dj,quadCoeffsMatrix[i,j,:],JLvalMatrix[i,j,:],JRvalMatrix[i,j,:] , JSize)
    
    cuda.syncthreads()
    #wait for all threads to finish
    
    #this seems like a bad way to do this,
    #if we are on the first thread 
    if k == 1:
        index =0
        
        while index < JSize:
            if index != 0:
                JLvalMatrix[i,j,index] = dj*JLvalMatrix[i,j,index-1]+JLvalMatrix[i,j,index]
            index += 1
    elif k == 2:
        index = 0
        while index < JSize:
            if index !=JSize-1:
                reverseIndex = JSize-index-1
                JRvalMatrix[i,j,reverseIndex-1] = dj*JRvalMatrix[i,j,reverseIndex] + JRvalMatrix[i,j,reverseIndex-1]
            index += 1
    
    cuda.syncthreads()
    
    wMatrix[i,j,k] = (JLvalMatrix[i,j,k] + JRvalMatrix[i,j,k])
    
    cuda.syncthreads()
    
    if k == 1:
        leftComm[len(uvalsMatrix) - i - 1] = wMatrix[i][0]
        rightComm[i]  = wMatrix[i][JSize-1]
        
    cuda.syncthreads()

@cuda.jit#('void(float32[:,:,:],float32[:,:,:],float32[:,:,:], float32[:,:,:],float32,float32[:,:],float32[:,:])')
def uUpdate2D(wMatrix, xvals ,A,B,alpha,a,b):
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    k = cuda.threadIdx.x
    
    wMatrix[i,j,k] = wMatrix[i,j,k]+A[i,j]*math.exp(-alpha*(xvals[i,j,k] - a[i,j] ) ) + B[i,j]*math.exp(-alpha *(b[i,j]  - xvals[i,j,k]))
    cuda.syncthreads()

@cuda.jit#('void(float32[:,:,:],float32[:,:,:],float32,float32,float32[:,:,:],float32)')
def uFinalize(uvalsMatrix, wMatrix, dt, time, uPrevMatrix , beta ): 
    i = cuda.blockIdx.x
    j = cuda.blockIdx.y
    k = cuda.threadIdx.x
    
    wMatrix[i,j,k] = uvalsMatrix[i,j,k] - .5*wMatrix[i,j,k]
    wMatrix[i,j,k] = 2*uvalsMatrix[i,j,k] - uPrevMatrix[i,j,k] - beta**2*wMatrix[i,j,k]
    uPrevMatrix[i,j,k] = uvalsMatrix[i,j,k]
    uvalsMatrix[i,j,k] = wMatrix[i,j,k]
    
    cuda.syncthreads()
