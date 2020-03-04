import numpy as np
from sklearn.neighbors import NearestNeighbors
from distances import *


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        if (strategy != 'my_own') & (strategy != 'brute') & (strategy != 'kd_tree') & (strategy != 'ball_tree'):
            raise TypeError('Type Error Strategy')
        self.strategy = strategy
        if (metric != 'euclidean') & (metric != 'cosine'):
            raise TypeError('Type Error Metric')
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        
    def fit(self, X, y):
        if self.strategy != 'my_own':
            self.model = NearestNeighbors(algorithm=self.strategy, metric=self.metric, n_neighbors=self.k)
            self.model.fit(X, y)
        self.train = X
        self.target = y
            
    def find_kneighbors(self, X, return_distance):
        if self.strategy != 'my_own':
            self.kneighbors = self.model.kneighbors(X,  n_neighbors=self.k, return_distance=return_distance)
        else:
            if self.metric == 'euclidean':
                self.kneighbors = euclidean_distance(X, self.train)
            else:
                self.kneighbors = cosine_distance(X, self.train)
            if return_distance:
                dist = np.sort(self.kneighbors)[:,:self.k]
            vv = np.arange(0, self.train.shape[0])
            vv = vv[np.newaxis, :]
            tmp = np.repeat(vv, repeats=X.shape[0], axis=0)
            ind = np.argsort(self.kneighbors, axis=1)
            for i,j in enumerate(ind):
                x = list(zip(tmp[i], j))
                x.sort()
                m,tmp[i] = zip(*x)
            self.kneighbors = tmp[:,:self.k]
            if return_distance:
                return(dist, self.kneighbors)
        return self.kneighbors
            
    def predict(self, X):
        ans = []
        for l in self.find_kneighbors(X, False):
            un_el, un_ind = np.unique(self.target[l], return_inverse=True)
            if un_el.shape[0] == 1:
                ans.append(un_el[0])
            else:
                count_el = np.bincount(un_ind)
                ans.append(zip(count_el,un_el).max()[1])
        return ans