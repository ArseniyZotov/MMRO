def calc_expectations(h, w, X, Q):
    import numpy as np

    def sum_k_columns_matrix(size, k):
        a = np.arange(size) - np.arange(size)[:, np.newaxis]
        return np.array((a > -k) & (a < 1), dtype=int)

    sum_cols = sum_k_columns_matrix(Q.shape[1], w).T
    sum_rows = sum_k_columns_matrix(Q.shape[0], h)
    prob_Q = sum_rows.dot(Q).dot(sum_cols)
    return (prob_Q * X)
