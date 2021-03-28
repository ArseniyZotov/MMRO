import time
import oracles
from scipy.special import expit
import numpy as np

class get_batch:
    def __init__(self, batch_size, file_name, vectorizer):
        self.batch_size = batch_size
        self.file_name = file_name
        self.vectorizer = vectorizer
    
    def __iter__(self):
        self.iter = pd.read_csv("mixed_data.csv", iterator=True, chunksize=self.batch_size, index_col=0)
        return self
    
    def __next__(self):
        try:
            comments = self.iter.get_chunk()
        except StopIteration:
            self.iter = pd.read_csv("mixed_data.csv", iterator=True, 
                                    chunksize=self.batch_size, index_col=0)
            comments = self.iter.get_chunk()
        comments.fillna("", inplace=True)
        return self.vectorizer.transform(comments["comment_text"])
    
class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, loss_function, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        max_iter - максимальное число итераций     
        
        **kwargs - аргументы, необходимые для инициализации   
        """
        if loss_function == 'binary_logistic':
            self.oracle_ = oracles.BinaryLogistic(**kwargs)
            
        self.step_alpha_ = step_alpha
        self.step_beta_ = step_beta
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.kwargs_ = kwargs
        
        
    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """
        Обучение метода по выборке X с ответами y
        
        X, X_val - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y, y_val - одномерный numpy array
        
        w_0 - начальное приближение в методе
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        history['accuracy']: list of floats, содержит значения accuracy на валидационной выборке на каждой итерации
        (0 для самой первой точки)
        """

        self.weights_ = w_0
        iter_num = 0
        
        history = {}
        last_func = 0
        cur_func = self.oracle_.func(X, y, self.weights_)
        
        if trace == True:
            history["time"] = [0]
            history["func"] = [cur_func]
            if X_val != None:
                y_pred = self.predict(X_val) 
                history["accuracy"] = [(y_pred==y_val).sum()/y_val.size]
                
        start_time = time.time()
        
        while iter_num < self.max_iter_ and \
                abs(last_func-cur_func) > self.tolerance_:
            last_func = cur_func
            iter_num += 1
            
            rate = self.step_alpha_ / iter_num ** self.step_beta_
            
            self.weights_ = self.weights_ - rate*self.oracle_.grad(X, y, self.weights_)
            
            cur_func = self.oracle_.func(X, y, self.weights_)
            
            if trace == True:
                history["time"].append(time.time() - start_time)
                history["func"].append(cur_func)
                start_time = time.time()
                if X_val != None:
                    y_pred = self.predict(X_val) 
                    history["accuracy"].append((y_pred==y_val).sum()/y_val.size)
                    
        if trace == True:
            return history
        
    def predict(self, X):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        answer = np.sign(X.dot(self.weights_))
        answer[answer == 0] = 1
        return answer
    
    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """
        answer = np.zeros((X.shape[0], 2))
        answer[:, 1] = expit(X.dot(self.weights_))
        answer[:, 0] = 1 - answer[:, 1]
        return answer
    
    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        return self.oracle_.func(X, y, self.weights_)
        
    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """
        return self.oracle_.grad(X, y, self.weights_)
    
    def get_weights(self):
        """
        Получение значения весов функционала
        """    
        return self.weights_


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        
        max_iter - максимальное число итераций (эпох)
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """
        
        if loss_function == 'binary_logistic':
            self.oracle_ = oracles.BinaryLogistic(**kwargs)
        
        self.random_seed_ = random_seed
        self.step_alpha_ = step_alpha
        self.step_beta_ = step_beta
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.kwargs_ = kwargs
        self.batch_size_ = batch_size
        
    def fit(self, X, y, w_0=None, trace=False, log_freq=1, X_val=None, y_val=None):
        """
        Обучение метода по выборке X с ответами y
        
        X, X_val - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y, y_val - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        history['accuracy']: list of floats, содержит значения accuracy на валидационной выборке на каждой итерации
        (0 для самой первой точки)
        """
        
        self.weights_ = w_0
        iter_num = 0
        
        history = {}
        last_func = 0
        cur_func = self.oracle_.func(X, y, self.weights_)
        obj_num = 0
        last_log = 0
        last_weights = w_0
        
        if trace == True:
            history['epoch_num'] = [0]
            history["time"] = [0]
            history["func"] = [cur_func]
            history['weights_diff'] = [0]
            if X_val != None:
                y_pred = self.predict(X_val) 
                history["accuracy"] = [(y_pred==y_val).sum()/y_val.size]
                
        start_time = time.time()
        
        while last_log < self.max_iter_ and \
                abs(last_func-cur_func) > self.tolerance_:
            
            indices_permutation = np.random.permutation(y.size)
            
            for j in range(0, y.size, self.batch_size_):
                indices = indices_permutation[j:j+self.batch_size_]
                obj_num += indices.size
                iter_num += 1

                rate = self.step_alpha_ / iter_num**self.step_beta_

                self.weights_ = self.weights_ - rate*self.oracle_.grad(X[indices], y[indices], self.weights_)
                
                if obj_num / y.size - last_log >= log_freq:
                    last_func = cur_func
                    cur_func = self.oracle_.func(X, y, self.weights_)
                    last_log = obj_num / y.size
                    
                    if trace == True:
                        history['epoch_num'].append(last_log)
                        history["time"].append(time.time() - start_time)
                        history["func"].append(cur_func)
                        dif_w = last_weights - self.weights_
                        history['weights_diff'].append(dif_w.dot(dif_w))
                        if X_val != None:
                            y_pred = self.predict(X_val) 
                            history["accuracy"].append((y_pred==y_val).sum()/y_val.size)
                            
                    start_time = time.time()
        
        if trace == True:
            return history
        
    def fit_special(self, X_iter, y, w_0=None, log_freq=1):
        self.weights_ = w_0
        iter_num = 0

        obj_num = 0
        last_log = 0
        last_weights = w_0
        y_begin = 0

        start_time = time.time()
        
        while last_log < self.max_iter_:

            X_train = next(X_iter)
            obj_num += X_train.shape[0]
            iter_num += 1
            y_train = y[y_begin:y_begin + X_train.shape[0]]
            y_begin += X_train.shape[0]
            if y_begin == y.size:
                y_begin = 0

            rate = self.step_alpha_ / iter_num**self.step_beta_

            self.weights_ = self.weights_ - rate*self.oracle_.grad(X_train, y_train, self.weights_)

            if obj_num / y.size - last_log >= log_freq:
                last_log = obj_num / y.size
                print(last_log)