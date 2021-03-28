def find_max_substring_occurrence(input_string):
    for i in range(len(input_string), 0, -1):
        if len(input_string) % i == 0:
            s = input_string[:len(input_string)//i] * i
            if s == input_string:
                return i
