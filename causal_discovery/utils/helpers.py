from itertools import chain, combinations


def append_value(array, i, j, value):
    """Append value to the list at array[i, j]"""
    if array[i, j] is None:
        array[i, j] = [value]
    else:
        array[i, j].append(value)


def powerset(L):
    """Return the powerset of L (list)"""
    s = list(L)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def list_union(L1, L2):
    """Return the union of L1 and L2 (lists)"""
    return list(set(L1 + L2))


def sort_dict_ascending(d, descending=False):
    """Sort dict by its value in ascending order"""
    dict_list = sorted(d.items(), key=lambda x: x[1], reverse=descending)
    return {dict_list[i][0]: dict_list[i][1] for i in range(len(dict_list))}
