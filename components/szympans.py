class Szympans:
    """
    Szympans is a parody of pandas. For the purpose of this classes, we are
    forbidden to use high-level libraries. This is a tool to load and
    do simple manipulations on data stored in txt files.
    """
    def __init__(self) -> None:
        pass

    @classmethod
    def read_txt(self, path: str, dlm: str) -> list:
        with open(path) as f:
            data = []
            for line in f:
                inner_list = [elmnt.strip() for elmnt in line.split(dlm)]
                for idx, _ in enumerate(inner_list):
                    inner_list[idx] = str(inner_list[idx])

                data.append(inner_list)

        return data

    @classmethod
    def nunique(self, data) -> dict:
        dict_with_counts = {}
        for i in range(len(data[0])):
            dict_with_counts[i] = len(set([x[i] for x in data]))

        return dict_with_counts

    @classmethod
    def count_values(self, data) -> dict:

        dict_of_dicts = {}
        for i in range(len(data[0])):
            list_to_count_occurences = ([x[i] for x in data])
            d = {x: list_to_count_occurences.count(x)
                 for x in list_to_count_occurences}
            dict_of_dicts[i] = d

        return dict_of_dicts
