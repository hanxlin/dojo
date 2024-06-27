
class Node:
    def __init__(self, data):
        self.val = data
        self.next = None


# ====== Helpers ======

def make_ll(ls):
    root = Node(ls[0])
    r = root
    for i in range(1, len(ls)):
        new_ndoe = Node(ls[i])
        r.next = new_ndoe
        r = new_ndoe
    return root

def tolist(node):
    out = []
    while node:
        out.append(node.val)
        node = node.next
    return out

# -----------------------


def reverse_ll(root):
    if not root.next:
        return root

    prev_root = root
    root = root.next
    root_next = root.next
    root.next = prev_root
    prev_root.next = None
    while root_next:
        prev_root = root
        root = root_next
        root_next = root_next.next
        root.next = prev_root
    return root


root = make_ll([1, 3, 5, 7, 9])
print('input:', tolist(root))
root = reverse_ll(root)
print('  >', tolist(root))

root = make_ll([1])
print('input:', tolist(root))
root = reverse_ll(root)
print('  >', tolist(root))

# NOTE
# A -> B -> C -> D
# root 
# tmp root tmp_next

# A <- B -> C -> D
# pev root next
#     pev root next