def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    gradient = np.zeros(w.size)
    for i in range(w.size):
        e_i = np.zeros(w.size)
        e_i[i] = 1
        f_1 = function(w)
        f_2 = function(w + eps*e_i)
        gradient[i] = (f_2-f_1) / eps
        
    return gradient