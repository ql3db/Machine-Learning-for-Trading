import numpy as np		 	 		 		 	 		  	 	 			  	 
import RTLearner as rt
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class BagLearner(object):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 	 #constructor	 		 	 		  	 	 			  	 
    def __init__(self, learner, bags, boost=False, verbose = False,**kwargs): 
        self.bags = bags
        #print("len(learner)",len(learner))
        self.boost = boost
        self.verbose = verbose
        self.leaf_size = kwargs.get('kwargs').get('leaf_size')
        #create "bags" amount of instances in learners
        self.isLin=1
        self.learners = [learner(leaf_size = self.leaf_size, verbose = self.verbose) for x in range(self.bags)]            
            
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def author(self):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        return 'qli385' # replace tb34 with your Georgia Tech username  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 	
    #training step
    def addEvidence(self,dataX,dataY): 
        #number in training set
        n = dataX.shape[0]

        i = 0
        while i < self.bags:
            #randomly select data
            dataXSelected, dataYSelected = self.random_select(n, dataX,dataY)
            #train data
            self.learners[i].addEvidence(dataXSelected, dataYSelected)
            i = i + 1
        
    
    def random_select(self, n, dataX, dataY):
        index = np.random.random(size=n)*n
        index = [int(x) for x in index]
        dataXSelected = dataX[index,:] 
        dataYSelected = dataY[index]
        return dataXSelected, dataYSelected
    
    def query(self, points):
        self.predY = np.ones((self.bags,points.shape[0]))
        i = 0
        while i < self.bags:
            #print(i)
            if (self.isLin):
                self.predY[i,:] = self.learners[i].query(points)
                #print("self.preY[",i,"]:", self.predY[i,:])
            else:
                self.predY[i,:] = self.learners[i].query(points)
            i = i+1
        self.predYMean = np.mean(self.predY, axis = 0)
        #print("predY.shape",self.predY.shape)
        return self.predYMean
    #.reshape(points.shape[0],)
            
                
    
    if __name__=="__main__":
        print("the secret clue is 'zzyzx'") 
        