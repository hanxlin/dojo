ls1 = [(1,1), (4, 6), (7,10)]
ls2 = [(1,8), (9, 10), [11,12]]

j = 0
merge_ct = 0
isolated_ct = 0
for i in range(len(ls1)-1):
    prv_st, prv_end = ls1[i]
    nxt_st, nxt_end = ls1[i+1]
    
    st, end = ls2[j]

    print(prv_end, nxt_st, st, end)
    if st <= prv_end:
        if end >= nxt_st:
            merge_ct += 1
            if end < nxt_end:
                j += 1
        
    else: # st > prv_end
        if end < nxt_st:
            isolated_ct += 1

        j += 1

print(merge_ct)
print(isolated_ct)