def replace_nan_to_means(X):
    import numpy as np
    mean_cols = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X_copy = X.copy()
    X_copy[inds] = mean_cols[inds[1]]
    return X_copy
