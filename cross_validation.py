#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from nearest_neighbors import *


# In[ ]:

def kfold(n, n_folds):
    k_list = []
    for i in range(0, n, round(n / n_folds + 0.5)):
        k_list.append((np.hstack((np.arange(0, i), np.arange(i + round(n / n_folds + 0.5), n))), np.arange(i, min(i + round(n / n_folds + 0.5), n))))
    if len(k_list) != n_folds:
        k_list.append((np.arange(0, n - 1), np.arange(n - 1,n)))
    return k_list

def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    acc = {}
    for key in k_list:
        acc[key] = np.array([])
    a = KNNClassifier(k=max(k_list), **kwargs)
    if cv == None:
        cv = kfold(X.shape[0], 3)
    for m,i in enumerate(cv):
        a.fit(X[i[0]], y[i[0]])
        dist, kneighbors = a.find_kneighbors(X[i[1]], True)
        for j,k in enumerate(k_list):
            ans = []
            dist_k = dist[:,:k]
            kneighbors_k = kneighbors[:,:k]
            for ind,l in enumerate(kneighbors_k):
                if a.weights:
                    count_el = np.bincount(y[i[0]][l].astype('int64'), weights = 1 / (dist_k[ind] + 0.00001))
                else:
                    count_el = np.bincount(y[i[0]][l].astype('int64'))
                ans.append(str(count_el.argmax()))
            acc[k] = np.append(acc[k], (np.array(ans).astype('int64') == np.array(y[i[1]]).astype('int64')).sum() / len(i[1]))

    return acc