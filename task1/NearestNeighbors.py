import Distances
import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNClassifier:
    def __init__(self, k = 5, strategy = 'my_own', metric = 'euclidean', weights = False, test_block_size = 10):
        self.k = k
        
        if (strategy not in ['my_own', 'brute', 'kd_tree', 'ball_tree']):
            raise Exception("Wrong 'strategy' param value")
        
        self.strategy = strategy
        
        if (metric not in ['euclidean', 'cosine']):
            raise Exception("Wrong 'metric' param value")
        
        self.metric = metric
        self.weights = weights
        self.testBlockSize = test_block_size
        
        if (self.strategy != 'my_own'):
            self.neigh = NearestNeighbors(self.k, algorithm = self.strategy)
 
    def fit(self, X, y):
        if (self.strategy == 'my_own'):
            self.X = X
            self.y = y
        else:
            self.neigh.fit(X, y)
    
    def find_kneighbors(self, X, return_distance = True):
        if (self.strategy == 'my_own'):
            
            res = []
            dists = []

            if (self.metric == 'euclidean'):
                dsModRes = Distances.euclidean_distance(X, self.X)
            else:
                dsModRes = Distances.cosine_distance(X, self.X)

            for i in range(len(dsModRes)):
                tmpDists = np.argsort(dsModRes[i])[:self.k]
                res.append(dsModRes[i][tmpDists])
                if (return_distance):
                    dists.append(tmpDists)
                    
            if (return_distance):
                return (res, dists)
            else:
                return (res)
            
        else:
            return self.neigh.kneighbors(X, return_distance = return_distance)

    def predict(self, X):
        print('predict')
