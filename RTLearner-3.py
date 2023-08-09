import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class RTLearner(object):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 	 #constructor	 		 	 		  	 	 			  	 
    def __init__(self, leaf_size = 1, verbose = False): 
        self.leaf_size = leaf_size
        self.verbose = verbose	  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def author(self):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        return 'qli385' # replace tb34 with your Georgia Tech username  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 	
    #training step
    def addEvidence(self,dataX,dataY):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Add training data to learner  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param dataX: X values of data to add  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param dataY: the Y training values  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        # newdataX filss with dataX, last column is 1 so get a constant 		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        #newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        #newdataX[:,0:dataX.shape[1]]=dataX  		  	   		     			  		 	
        #corner case
        #when there is only 1 line of data
        #DTree
            #col1: the number of the factor used, -1 indicates it's a leaf
            #col2: Y_predict
            #col3: relative location of left node, 0 indicates it has no left tree
            #col4: relative location of right node, 0 indicates it has no right tree
        self.DTree = np.array([]).reshape(0,4)
        #if (dataX.shape[0] < self.leaf_size): 
            #print("dataX.shape",dataX.shape)
            #return self.DTree
        if (dataX.shape[0] <= self.leaf_size): 
            #print("dataY",dataY)
            self.DTree = (np.array([-1, np.mean(dataY), 0, 0])).reshape(1,4)
            return self.DTree
        elif np.std(dataY) == 0.0:
            #print("same")
            self.DTree = (np.array([-1, np.mean(dataY), 0, 0])).reshape(1,4)
            return self.DTree
        else:
            #determine a random feature i to split on using correlation
            #also make sure correlation is not nan
            best = int(np.random.random()*(dataX.shape[1]))
            while np.isnan(np.corrcoef(dataX[:,best],dataY)[0,1]):
                best = int(np.random.random()*(dataX.shape[1] ))
                #print("best:", best)
            
            #choose the median of the best feature
            splitVal = np.median(dataX[:,best])
            if (splitVal ==np.max(dataX[:,best]) or splitVal ==np.min(dataX[:,best])):
                splitVal = np.mean(dataX[:,best])
            #print(splitVal)
            
            #build subtrees
            newData = np.append(dataX, dataY.reshape(dataY.shape[0],1),axis=1)
            #drop duplicates
            newData = np.unique(newData, axis=0)
            #print("newData shape:",newData.shape)
            #print("DataX:" ,dataX.shape)
            dataLeft = newData[newData[:,best]<=splitVal]
            dataRight = newData[newData[:,best]>splitVal]
            leftTree = self.addEvidence(dataLeft[:,0:-1],dataLeft[:,-1])
            rightTree = self.addEvidence(dataRight[:,0:-1],dataRight[:,-1])
            #print("ledtTree.shape",leftTree.shape[0])
            #print("leftTree",leftTree)
            #leftTree = leftTree.reshape(int(leftTree.shape[0]/4),4)
            #rightTree = rightTree.reshape(int(rightTree.shape[0]/4),4)
            
            #print("rightTree",rightTree)
            root = (np.array(([best, splitVal, 1, leftTree.shape[0]+1]))).reshape(1,4)
            #print("dimensions",root.shape,leftTree.shape,rightTree.shape)
            self.DTree = np.concatenate((root, leftTree,rightTree))
            #print("DTree shape:",DTree.shape)
            #DTree = DTree.reshape(int(DTree.shape[0]/4),4)
            #print("DTree shape2:",DTree.shape)
            return self.DTree
            #save the tree
            
        # build and save the model  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        #linear regressions
        #self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY, rcond=None)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def query(self,points):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @summary: Estimate a set of test points given the model we built.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @param points: should be a numpy array with each row corresponding to a specific query.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        @returns the estimated values according to the saved model.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        #return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1] 
        
        n = points.shape[0]
        predY = np.ones(n)
        i = 0
        while i < n:
            dataX = points[i,:]
            y = self.predict(dataX)
            predY[i] = y
            i = i+1
        return predY
    
    def predict(self, dataX):
        if (len(dataX.shape) > 1 and self.verbose ==True):
            print("Can only predict data for once each time")
        curNode = 0
        while int(self.DTree[curNode,0]) != -1:
            #if the target value <= splitVal
            best = int(self.DTree[curNode, 0])
            #print("best",best)
            if dataX[best] <= self.DTree[curNode,1]:
                left = int(self.DTree[curNode,2])
                curNode = curNode + left
                #print("Left:",left)
            else: 
                right = int(self.DTree[curNode,3])
                curNode = curNode + right
                #print("Right:",right)
            #print(curNode)
        #now it goes to a row that 1st column is -1
        predY = self.DTree[curNode,1]
        return predY
            
            
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__=="__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("the secret clue is 'zzyzx'")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
