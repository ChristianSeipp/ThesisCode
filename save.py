        ############################################# X-Axis ########################################
        hwMatrix = hwMatrix.T
        
        hPlist  = np.array([ quad.P_ for quad in self.xQuadWeights] , dtype = np.float32)
        hQlist  = np.array([ quad.Q_ for quad in self.xQuadWeights] , dtype = np.float32)
        hRlist  = np.array([ quad.R_ for quad in self.xQuadWeights] , dtype = np.float32)
        
        hvjsqrinvList = np.array([quad.vjsqrinv_ for quad in self.xQuadWeights] , dtype = np.float32)
        hdjList       = np.array([quad.dj_ for quad in self.xQuadWeights] , dtype = np.float32)
        dwMatrix = cuda.to_device(hwMatrix)
        
        dPlist = cuda.to_device(hPlist)
        dQlist = cuda.to_device(hQlist)
        dRlist = cuda.to_device(hRlist)
        
        dvjsqrinvList = cuda.to_device(hvjsqrinvList)
        ddjList       = cuda.to_device(hdjList)
        
        hWYMatrix = np.zeros_like(hwMatrix)
        dWYMatrix = cuda.to_device(hWYMatrix)

        calcLocalU[self.numBlocks,self.TPB](dwMatrix, dnDomains , dPlist, dQlist , 
                                            dRlist , dvjsqrinvList , ddjList , 
                                            dquadCoeffs , dWYMatrix, dJLvals , 
                                            dJRvals , dleftComm, drightComm, dJSize )
        
        hWYMatrix = dJRvals.copy_to_host()

        hWYMatrix = hWYMatrix.T
        #print(hWYMatrix)
        expAlphaList  = np.array([ quad.expAlphaDist_ for quad in self.xQuadWeights] , dtype = np.float32)
        dAVals   = cuda.to_device(np.zeros_like(hrightComm))
        dBVals   = cuda.to_device(np.zeros_like(hleftComm))
        
        updateTransmissionCoeffs[self.numBlocks[0],2](dleftComm , drightComm , expAlphaList , self.xDamping_ , 
                                                      dAVals ,dBVals , dnDomains)

        uUpdate2D[self.numBlocks,self.TPB](dWYMatrix , self.yMeshed ,  dAVals , dBVals , self.alpha_, 
                                           self.beta_, self.xa_ , self.xb_ ,dJSize)
        
        hWYMatrix = dWYMatrix.copy_to_host()

        hWYMatrix = hWYMatrix.T
        #print(hWYMatrix)

        
        dWYMatrix = cuda.to_device(hWYMatrix)
        
        huPrev = self.uprev_
        duPrev = cuda.to_device(huPrev)

        finalize2D[self.numBlocks,self.TPB](duvals, duPrev, dWYMatrix , self.beta_ , dJSize )
        
        self.ucurr_ = duvals.copy_to_host()
        self.uprev_ = duPrev.copy_to_host()
