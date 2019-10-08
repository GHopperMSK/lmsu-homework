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
        
        self.neigh = NearestNeighbors(self.k, algorithm = self.strategy)
 
    def fit(self, X, y):
        self.neigh.fit(X, y)
        #self.X = X
        #self.y = y
    
    def find_kneighbors(self, X, return_distance = True):
        
        if (self.strategy == 'my_own'):
            return 1
        else:
            #neigh.fit(self.X, self.y)
            return self.neigh.kneighbors(X)

    def predict(self, X):
        print('predict')
