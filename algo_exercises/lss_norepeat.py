# find the length of longest substring without repeating characters

# ex. 1
# s = "abcabcbb"
# output 3

# ex. 2
# s = "bbbbb"
# output 1

# ex. 3
# s = "pwwkew"
# output 3


# NOTE: core of the problem is basically iterating all windows that contain no repeating chars.
# but using a sliding window smartly allows us to not have to iterate over all such windows.

def lls_norepeat(s):
    i, j = 0, 0
    records = {}
    max_len = 0
    while i < len(s) and j < len(s):
        while j < len(s):
            ch = s[j]
            if ch in records:
                break
            records[ch] = j
            j += 1
        # ch is in records, ie. we encounter a repeated
        # get the current window len
        cur_len = j - i # subtraction excludes one of the ends, as desired
        # update max len
        if cur_len > max_len:
            max_len = cur_len
        # get the saved idx of ch as seen previously
        new_i = records[ch] + 1
        # before sliding i to right, remove all recorded ch outside the new window
        for k in range(i, new_i):
            del records[s[k]]
        i = new_i

    return max_len


print(lls_norepeat("a"))
print(lls_norepeat("abcabcbb"))
print(lls_norepeat("bbbbb"))
print(lls_norepeat("pwwkew"))
print(lls_norepeat("abba"))