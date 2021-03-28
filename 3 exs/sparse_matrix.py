class CooSparseMatrix:

    def __init__(self, ijx_list, shape):
        if len(shape) != 2:
            raise TypeError()
        for ijx in ijx_list:
            if len(ijx) != 3 or ijx[0] >= shape[0] or \
                    ijx[1] >= shape[1] or ijx[0] < 0 or ijx[1] < 0:
                raise TypeError()
        ijx_list.sort()
        for i in range(len(ijx_list) - 1):
            if ijx_list[i][:2] == ijx_list[i + 1][:2]:
                raise TypeError()
        self.dict = {(x[0], x[1]): x[2] for x in ijx_list if x[2] != 0}
        self._shape = shape

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= self._shape[0] or item < 0:
                raise TypeError()
            ijx_list = [(item, j, self.dict[(item, j)])
                        for j in range(self._shape[1]) if (item, j) in self.dict]
            return CooSparseMatrix(ijx_list, (1, self._shape[1]))

        if len(item) == 2:
            if item[0] >= self._shape[0] or item[0] < 0 or \
                    item[1] >= self._shape[1] or item[1] < 0:
                raise TypeError()
            if (item[0], item[1]) in self.dict:
                return self.dict[(item[0], item[1])]
            return 0
        raise TypeError()

    def __setitem__(self, key, value):
        if key[0] >= self._shape[0] or key[0] < 0 or \
                key[1] >= self._shape[1] or key[1] < 0 or len(key) != 2:
            raise TypeError()
        if value != 0:
            self.dict[key] = value
        elif key in self.dict:
            self.dict.pop(key)

    def __add__(self, other):
        if self._shape != other._shape:
            raise TypeError()
        new_dict = self.dict.copy()
        for key, value in other.dict.items():
            if key in new_dict:
                new_dict[key] += value
                if new_dict[key] == 0:
                    del new_dict[key]
            else:
                new_dict[key] = value
        new_matrix = CooSparseMatrix(ijx_list=[], shape=self._shape)
        new_matrix.dict = new_dict
        return new_matrix

    def __sub__(self, other):
        if self._shape != other._shape:
            raise TypeError()
        new_dict = self.dict.copy()
        for key, value in other.dict.items():
            if key in new_dict:
                new_dict[key] -= value
                if new_dict[key] == 0:
                    del new_dict[key]
            else:
                new_dict[key] = -value
        new_matrix = CooSparseMatrix(ijx_list=[], shape=self._shape)
        new_matrix.dict = new_dict
        return new_matrix

    def __mul__(self, other):
        new_matrix = CooSparseMatrix(ijx_list=[], shape=self._shape)
        if other == 0:
            return new_matrix
        new_dict = self.dict.copy()
        for key in new_dict.keys():
            new_dict[key] *= other
        new_matrix = CooSparseMatrix(ijx_list=[], shape=self._shape)
        new_matrix.dict = new_dict
        return new_matrix

    def __rmul__(self, other):
        new_matrix = CooSparseMatrix(ijx_list=[], shape=self._shape)
        if other == 0:
            return new_matrix
        new_dict = self.dict.copy()
        for key in new_dict.keys():
            new_dict[key] *= other
        new_matrix = CooSparseMatrix(ijx_list=[], shape=self._shape)
        new_matrix.dict = new_dict
        return new_matrix

    def set_shape(self, value):
        if isinstance(value, tuple) and len(value) == 2 and \
                isinstance(value[0], int) and isinstance(value[1], int) and \
                value[0] > 0 and value[1] > 0 and \
                value[0] * value[1] == self._shape[0] * self._shape[1]:
            new_dict = {((key[0] * self._shape[1] + key[1]) // value[1],
                         (key[0] * self._shape[1] + key[1]) % value[1]): val
                        for key, val in self.dict.items()}
            self._shape = value
            self.dict = new_dict
        else:
            raise TypeError()

    def get_shape(self):
        return self._shape

    shape = property(get_shape, set_shape, None, "no doc")

    def set_T(self, value):
        raise AttributeError()

    def get_T(self):
        new_matrix = CooSparseMatrix(ijx_list=[],
                                     shape=(self._shape[1], self._shape[0]))
        new_dict = {(key[1], key[0]): value for key, value in self.dict.items()}
        new_matrix.dict = new_dict
        return new_matrix

    T = property(get_T, set_T, None, "no doc")
