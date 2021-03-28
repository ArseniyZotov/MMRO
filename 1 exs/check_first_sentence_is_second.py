def check_first_sentence_is_second(s1, s2):
    import collections
    c1 = collections.Counter()
    c2 = collections.Counter()
    for word in s1.split():
        c1[word] += 1
    for word in s2.split():
        c2[word] += 1
    c1.subtract(c2)
    for word in c1:
        if c1[word] < 0:
            return False
    return True
