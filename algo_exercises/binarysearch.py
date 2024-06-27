# sorted list [1, 4, 7, 8, 9, 15, 20]
# find idx of 9
# return -1 if not found

def binsearch(ls:list[int], target):
    # ls is sorted list of int
    print(ls, target)
    st, ed = 0, len(ls)
    i = ed // 2
    itr = 0
    while i >= st and i < ed:
        print('jumpped to', i)
        val = ls[i]
        if val == target:
            return i
        if target > val:
            i += (ed - i) // 2
        else:
            i = (i - st) // 2

    return -1


print(' >> final answer', binsearch([1, 4, 5, 7, 8, 9, 15, 20], 8))
print(' >> final answer', binsearch([1, 4, 5, 7, 8, 9, 15, 20], 15))
print(' >> final answer', binsearch([1, 4, 5, 7, 8, 9, 15, 20], 20))
print(' >> final answer', binsearch([1, 4, 5, 7, 8, 9, 15, 20], 1))