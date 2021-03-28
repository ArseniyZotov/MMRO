def get_nonzero_diag_product(X):
    import numpy as np
    non_zero_diag = np.diagonal(X)[np.diagonal(X) != 0]
    if non_zero_diag.size:
        return np.prod(non_zero_diag)
    return None
