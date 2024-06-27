# given unsort ls [5, 4, 8, 1, 9, 2]
# return a sorted list [1, 2, 4, 5, 8, 9]

def _merge(l, r):
    i, j = 0, 0 
    out = []
    while i < len(l) and j < len(r):
        if l[i] < r[j]:
            out.append(l[i])
            i += 1
        else:
            out.append(r[j])
            j += 1
    # check remaining in l or r
    if i < len(l):
        out.extend(l[i:])
    elif j < len(r):
        out.extend(r[j:])
    return out

def _mergesort(ls, st, mid, ed):
    if ed - st <= 1:
        return ls[st:ed]

    # recurs on left
    new_mid =  st + (mid - st) // 2
    l_sorted = _mergesort(ls, st, new_mid, mid)

    # recurs on right
    new_mid =  mid + (ed - mid) // 2
    r_sorted = _mergesort(ls, mid, new_mid, ed)

    return _merge(l_sorted, r_sorted)

def mergesort(ls):
    st, ed = 0, len(ls)
    return _mergesort(ls, st, ed // 2, ed) 


print(mergesort([5, 4, 8, 1, 9, 2]))
print(mergesort([5]))
print(mergesort([5, 4]))
print(mergesort([1, 2, 3, 4, 5]))
print(mergesort([]))