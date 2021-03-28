import numpy as np
from sklearn.neighbors import NearestNeighbors


def euclidean_distance(X, Y):
    norm_x = (X * X).sum(axis=1)
    norm_y = (Y * Y).sum(axis=1)
    scalar_matrix = X.dot(Y.T)
    return np.sqrt(norm_x[:, np.newaxis] + norm_y - 2 * scalar_matrix)


def cosine_distance(X, Y):
    norm_x = (X * X).sum(axis=1)
    norm_y = (Y * Y).sum(axis=1)
    scalar_matrix = X.dot(Y.T)
    return 1 - scalar_matrix / (np.sqrt(norm_x[:, np.newaxis] * norm_y))


class KNNClassifier:

    def __init__(self, k=1, strategy="my_own", metric="euclidean",
                 weights=False, test_block_size=3):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size

    def fit(self, X, y):
        self.train_labels = y
        if self.strategy == "my_own":
            self.train_data = X
        else:
            self.classifier = NearestNeighbors(n_neighbors=self.k,
                                               algorithm=self.strategy,
                                               metric=self.metric)
            self.classifier.fit(X, y)

    def find_kneighbors(self, X, return_distance):
        k_neigh_dist = np.zeros((X.shape[0], self.k))
        k_neigh_indices = np.zeros((X.shape[0], self.k), dtype=int)
        if self.strategy == "my_own":
            for i in range(0, X.shape[0], self.test_block_size):
                if self.metric == "euclidean":
                    distances = euclidean_distance(X[i:i + self.test_block_size],
                                                   self.train_data)
                else:
                    distances = cosine_distance(X[i:i + self.test_block_size],
                                                self.train_data)
                k_indices = np.argpartition(distances, range(self.k), axis=1)[:, :self.k]
                for j in range(k_indices.shape[0]):
                    k_neigh_dist[i + j] = distances[j][k_indices[j]]
                k_neigh_indices[i:i + self.test_block_size] = k_indices
        else:
            k_neigh_dist, k_neigh_indices = self.classifier.kneighbors(X, self.k, True)
        if return_distance:
            return k_neigh_dist, k_neigh_indices
        else:
            return k_neigh_indices

    def predict(self, X):
        predicted_labels = np.zeros(X.shape[0], dtype=int)
        weights = np.ones((X.shape[0], self.k))
        if self.weights:
            weights, k_indices = self.find_kneighbors(X, True)
            weights = 1 / (weights + 1e-5)
        else:
            k_indices = self.find_kneighbors(X, False)
        for i in range(X.shape[0]):
            k_labels = self.train_labels[k_indices[i]]
            predicted_labels[i] = np.argmax(np.bincount(k_labels, weights[i]))
        return predicted_labels


def kfold(n, n_folds):
    train_test_split = list()
    size = n // n_folds + 1
    for i in range(n % n_folds):
        test_begin = i * size
        train = np.concatenate((np.arange(0, test_begin), 
                               np.arange(test_begin+size, n)))
        test = np.arange(test_begin, test_begin+size)
        train_test_split.append((train, test))
    begin = (n % n_folds) * size
    size -= 1
    for i in range(n_folds - n % n_folds):
        test_begin = begin + i * size
        train = np.concatenate((np.arange(0, test_begin), 
                               np.arange(test_begin+size, n)))
        test = np.arange(test_begin, test_begin+size)
        train_test_split.append((train, test))
    return train_test_split


def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 3)
    score_dict = {}
    cv_len = len(cv)
    for k in k_list:
        score_dict[k] = np.zeros(cv_len)
    classifier = KNNClassifier(k=k_list[-1], **kwargs)
    for i in range(cv_len):
        train_ind, test_ind = cv[i]
        train_data = X[train_ind]
        train_labels = y[train_ind]
        test_data = X[test_ind]
        test_labels = y[test_ind]
        classifier.fit(train_data, train_labels)

        k_max_weights = np.ones((test_data.shape[0], k_list[-1]))
        if classifier.weights:
            k_max_weights, k_max_indices = classifier.find_kneighbors(test_data, True)
            k_max_weights = 1 / (k_max_weights + 1e-5)
        else:
            k_max_indices = classifier.find_kneighbors(test_data, False)

        for k in k_list:
            predicted_labels = np.zeros(test_data.shape[0], dtype=int)
            weights = k_max_weights[:, :k]
            k_indices = k_max_indices[:, :k]
            for j in range(test_data.shape[0]):
                k_labels = train_labels[k_indices[j]]
                predicted_labels[j] = np.argmax(np.bincount(k_labels, weights[j]))
            if score == "accuracy":
                score_dict[k][i] = np.sum(predicted_labels == test_labels) / test_labels.size
    return score_dict
