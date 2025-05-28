class Node:
    def __init__(self, data):
        self.data = data
        self.right = None
        self.left = None
        self.height = 1  # Height of the node


def getHeight(root):
    return root.height if root else 0


def getBalance(root):
    return getHeight(root.left) - getHeight(root.right) if root else 0


def rotateLeft(p):
    c = p.right
    p.right = c.left
    c.left = p

    p.height = 1 + max(getHeight(p.left), getHeight(p.right))
    c.height = 1 + max(getHeight(c.left), getHeight(c.right))

    return c 


def rotateRight(p):
    c = p.left
    p.left = c.right
    c.right = p

    p.height = 1 + max(getHeight(p.left), getHeight(p.right))
    c.height = 1 + max(getHeight(c.left), getHeight(c.right))

    return c  


def insert(root, data):
    if not root:
        return Node(data)

    if data < root.data:
        root.left = insert(root.left, data)
    elif data > root.data:
        root.right = insert(root.right, data)
    else:
        return root 

    root.height = 1 + max(getHeight(root.left), getHeight(root.right))

    balance = getBalance(root)

    # Case 1: Left Left (LL)
    if balance > 1 and data < root.left.data:
        print("----LL type----")
        return rotateRight(root)

    # Case 2: Right Right (RR)
    if balance < -1 and data > root.right.data:
        print("----RR type----")
        return rotateLeft(root)

    # Case 3: Left Right (LR)
    if balance > 1 and data > root.left.data:
        print("----LR type----")
        root.left = rotateLeft(root.left)
        return rotateRight(root)

    # Case 4: Right Left (RL)
    if balance < -1 and data < root.right.data:
        print("----RL type----")
        root.right = rotateRight(root.right)
        return rotateLeft(root)

    return root


def preOrder(root):
    if root is not None:
        print("%2d" % root.data, end=" ")
        preOrder(root.left)
        preOrder(root.right)


def display(root, msg):
    print(msg, end='')
    preOrder(root)
    print()


if __name__ == "__main__":
    root = None
    #datas = [35, 18, 7, 26, 3, 22, 30, 12, 26, 68, 99]
    datas = [7,10,4,6,5]

    for data in datas:
        root = insert(root, data)
        display(root, '[Insert %2d] : ' % data)
    print()
