import numpy as np

def euclidean_distance(X, Y):
    d = len(X[0])
    res = np.zeros(len(X)*len(Y)).reshape(len(X), len(Y))
    for x in range(len(X)):
        for y in range(len(Y)):
            sum = 0
            for f in range(len(Y[y])):
                sum += (X[x][f] - Y[y][f])**2
            res[x][y] = np.sqrt(sum)
    return res;

def cosine_distance(X, Y):
    sumyy = (Y**2).sum(1)
    sumxx = (X**2).sum(1, keepdims=1)
    sumxy = X.dot(Y.T)
    return 1 - (sumxy/np.sqrt(sumxx))/np.sqrt(sumyy)
