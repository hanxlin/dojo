# given [a, b, c, d]
# return all partitions of the set
# [], [a], [b], [c], [d], [a,b], [a,c], [a,d], [b,c], [b,c], [c,d], [a,b,c], [a,b,d], [a, c, d], [b,c,d], [a,b,c,d]


# [a], [a,b], [a,c], [a,d], [a,b,c], [a,b,d], [a, c, d], [a,b,c,d]
# [], [b], [c], [d], [b,c], [b,d], [b,c,d], [c, d]


def _allpart(ls, st):
    if len(ls) - st == 0:
        return [[]]

    item = ls[st]
    subsets = _allpart(ls, st+1)

    out = [[item] + subset for subset in subsets]
    out.extend(subsets)

    return out

def allpart(ls):
    return _allpart(ls, 0)

ex1 = allpart(['a', 'b', 'c', 'd'])
print(ex1)
print(len(ex1))
