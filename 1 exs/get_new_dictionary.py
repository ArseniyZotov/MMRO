def get_new_dictionary(input_dict_name, output_dict_name):
    import collections
    f_input = open(input_dict_name, "r")
    n = int(f_input.readline())
    new_dict = collections.defaultdict(list)
    for i in range(n):
        s = f_input.readline().strip()
        s = "".join(s.split())
        s = s.split("-")
        human_word = s[0]
        drag_word = s[1].split(",")
        for word in drag_word:
            new_dict[word].append(human_word)
    word_list = list(new_dict)
    word_list.sort()
    with open(output_dict_name, "w") as out_f:
        out_f.write(str(len(word_list)) + "\n")
        for word in word_list:
            out_f.write(word + " - "
                        + ", ".join(sorted(new_dict[word])) + "\n")
    f_input.close()
