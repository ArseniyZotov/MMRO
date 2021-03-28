def encode_rle(x):
    import numpy as np
    x_shift = np.roll(x, 1)
    mask = x_shift != x
    mask[0] = True
    unique_num = x[mask]
    ind_start = np.where(mask)[0]
    ind_end = np.roll(ind_start, -1)
    ind_end[-1] = x.size 
    unique_len = ind_end - ind_start
    return (unique_num, unique_len)
