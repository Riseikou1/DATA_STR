class Node :
    def __init__(self,data):
        self.data = data
        self.right = None
        self.left = None
        self.height = 1

def getHeight(root):
    return root.height if root else 0

def getBalance(root):
    return getHeight(root.left) - getHeight(root.right) if root else 0

def insert(root,data):
    if not root :
        return Node(data)
    
    if data < root.data :
        root.left = insert(root.left,data)
    
    elif data > root.data :
        root.right = insert(root.right,data)

    else :
        return root
    
    root.height = 1 + max(getHeight(root.left), getHeight(root.right))

    balance = getBalance(root)

    if balance > 1 and data < root.left.data :
        return rotateRight(root)
    if balance < -1 and data > root.right.data :
        return rotateLeft(root)
    if balance > 1 and data > root.left.data :
        root.left = rotateLeft(root.left)
        return rotateRight(root)
    if balance < -1 and data < root.right.data :
        root.right = rotateRight(root.right)
        return rotateLeft(root)
    
    if balance > 1 and data < root.left.data :
        return rotateRight(root)
    if balance < -1 and data > root.right.data :
        return rotateLeft(root)
    if balance > 1 and data > root.left.data :
        root.left = rotateLeft(root.left)
        return rotateRight(root)
    if balance < -1 and data < root.right.data :
        root.right = rotateRight(root.right)
        return rotateLeft(root)

    return root

def rotateRight(p):
    c = p.left
    p.left = c.right
    c.right = p

    p.height = 1 + max(getHeight(p.left), getHeight(p.right))
    c.height = 1 + max(getHeight(c.left), getHeight(c.right))
    return c

def rotateLeft(p):
    c = p.right
    p.right = c.left
    c.left = p

    p.height = 1 + max(getHeight(p.left), getHeight(p.right))
    c.height = 1 + max(getHeight(c.left), getHeight(c.right))
    return c



