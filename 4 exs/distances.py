import numpy as np

def euclidean_distance(X, Y):
    norm_x = (X*X).sum(axis=1)
    norm_y = (Y*Y).sum(axis=1)
    scalar_matrix = X.dot(Y.T)
    return np.sqrt(norm_x[:, np.newaxis] + norm_y - 2*scalar_matrix)
    
def cosine_distance(X, Y):
    norm_x = (X*X).sum(axis=1)
    norm_y = (Y*Y).sum(axis=1)
    scalar_matrix = X.dot(Y.T)
    return 1 - scalar_matrix/(np.sqrt(norm_x[:, np.newaxis]*norm_y))
