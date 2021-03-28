def find_word_in_circle(circle, word):
    circle_len = len(circle)
    if circle_len == 0:
        return -1
    circle = circle * (len(word)//circle_len + 2)
    for i in range(circle_len):
        if circle.startswith(word, i):
            return i, 1
    circle = circle[::-1]
    for i in range(circle_len):
        if circle.startswith(word, i):
            return circle_len - i - 1, -1
    return -1
