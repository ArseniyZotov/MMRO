def WordContextGenerator(words, window_size):
    length = len(words)
    for i in range(length):
        begin = max(0, i - window_size) 
        end = min(length, i + window_size + 1)
        for j in range(begin, end):
            if i != j:
                yield words[i], words[j]
