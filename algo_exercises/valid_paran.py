# l, m, n are number of '()', '<>', '{}' pairs respectively.
# Return all valid permutations of parentheses given l, m, n

lookup = {
    '(': ')',
    '<': '>',
    '{': '}'
}

def _rec(l, m, n, stack, accum, ans):
    # base case
    if l == 0 and m == 0 and n == 0:
        while len(stack) > 0:
            accum += lookup[stack.pop()]
        ans.append(accum)

    # all the options as branches
    # each branch must have its own state, ie. stack needs to be copied
    if len(stack) > 0:
        stack_cp = stack[:]
        op = stack_cp.pop()
        _rec(l, m, n, stack_cp, accum+lookup[op], ans)
    if l > 0:
        stack_cp = stack[:]
        stack_cp.append('(')
        _rec(l-1, m, n, stack_cp, accum+'(', ans)
    if m > 0:
        stack_cp = stack[:]
        stack_cp.append('<')
        _rec(l, m-1, n, stack_cp, accum+'<', ans)
    if n > 0:
        stack_cp = stack[:]
        stack_cp.append('{')
        _rec(l, m, n-1, stack_cp, accum+'{', ans)

def solve(l, m, n):
    ans = []
    _rec(l, m, n, [], '', ans)
    return ans

# test
print(solve(0,0,0))
print(solve(1,0,0))
print(solve(1,1,0))
print(solve(1,0,1))
print(solve(1,1,1))
