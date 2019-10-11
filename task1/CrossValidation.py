import NearestNeighbors
import numpy as np

def kfold(n, n_folds):
    partSize = n // n_folds
    indexes = [i for i in range(n)]
    np.random.shuffle(indexes)
    
    res = []
    for i in range(n_folds):
        test = indexes[partSize*i:partSize*(i+1)]
        valid = indexes[0:partSize*i] + indexes[partSize*(i+1):]
        res.append((test, valid))

    return res
    
    
def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    accuracy = {}
    for k in k_list:
        accuracy[k] = []
        
        classifier = NearestNeighbors.KNNClassifier(
            k = k, 
            strategy = 'my_own', 
            metric = 'euclidean', 
            weights = False,    
            test_block_size = 10
        )

        for c in cv:
            tmpY = y[c[1]]
            classifier.fit(X[c[1]], tmpY)
            res = classifier.find_kneighbors(X[c[0]], True)

            tmpAccuracy = 0
            for r in range(len(res[1])):
                for rr in range(len(res[1][r])):
                    if (tmpY[r] == tmpY[res[1][r][rr]]):
                        tmpAccuracy += 1

            accuracy[k].append(tmpAccuracy / len(X))
            
    return accuracy
 
