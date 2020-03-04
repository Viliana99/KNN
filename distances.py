import numpy as np


def euclidean_distance(X, Y):
    return np.sqrt(np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T))

def cosine_distance(X, Y):
    normalise_Y = Y / np.sqrt(np.apply_along_axis(func1d=np.sum, axis=1, arr=Y ** 2))[:, np.newaxis]
    normalise_X = X / np.sqrt(np.apply_along_axis(func1d=np.sum, axis=1, arr=X ** 2))[:, np.newaxis]
    res = np.dot(normalise_X, normalise_Y.T)
    return 1 - res