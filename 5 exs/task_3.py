import numpy as np

def get_indices(sequence, indices):
    if isinstance(sequence, list):
        return [sequence[i] for i in indices]
    return sequence[indices]

def BatchGenerator(list_of_sequences, batch_size, shuffle=False):
    if len(list_of_sequences) != 0:
        length = len(list_of_sequences[0])
        if shuffle:
            indices_permutation = np.random.permutation(length)
        else:
            indices_permutation = np.arange(length)
        for j in range(0, length, batch_size):
            indices = indices_permutation[j:j+batch_size]
            yield [get_indices(seq, indices) for seq in list_of_sequences]
