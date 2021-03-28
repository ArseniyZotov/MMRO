def get_max_before_zero(x):
    import numpy as np
    diag_shift = np.diag(np.ones((x.size - 1)), 1)
    mask = np.array(x == 0, dtype=int).dot(diag_shift)
    mask = np.array(mask, dtype=bool)
    if x[mask].size:
        return x[mask].max()
    return None
