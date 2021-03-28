import numpy as np

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

class RleSequence:
    
    def __init__(self, input_sequence):
        self.size = input_sequence.size
        self.num_, self.len_ = encode_rle(input_sequence)
        
    def __getitem__(self, item):
        if isinstance(item, slice):
            now = item.start
            stop = item.stop
            step = item.step
            if step == None:
                step = 1
            if now == None:
                now = 0
            if stop == None:
                stop = self.size
            if now < 0:
                now = self.size + now
            if stop < 0:
                stop = self.size + stop
            if now >= self.size or now < 0:
                raise IndexError
            stop = min(stop, self.size)
            
            ans = np.arange(now, stop, step, dtype=self.num_.dtype)
            i_num = 0
            cur_len = self.len_[0]
            
            for i in range(ans.size):
                while ans[i] >= cur_len:
                    i_num += 1
                    cur_len += self.len_[i_num]
                ans[i] = self.num_[i_num]
                now += step
                
            return ans
                    
        else:
            if item < 0:
                item = self.size + item
            if item >= self.size or item < 0:
                raise IndexError
            cur_len = 0
            for i_num in range(self.len_.size):
                cur_len += self.len_[i_num]
                if item < cur_len:
                    return self.num_[i_num]
                
    def __iter__(self):
        self.i_ = 0
        self.i_num_ = 0
        return self
    
    def __next__(self):
        if self.i_num_ == self.num_.size:
            raise StopIteration
        ans = self.num_[self.i_num_]
        self.i_ += 1
        if self.len_[self.i_num_] == self.i_:
            self.i_num_ += 1
            self.i_ = 0
        return ans
    
    def __contains__(self, item):
        return np.any(self.num_ == item)

