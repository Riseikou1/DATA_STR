class Node :
    def __init__(self,data):
        self.data = data
        self.right = None
        self.left = None
        self.height = 1

def getHeight(root):
    return root.height if root else 0

def getBalance(root):
    return getHeight(root.left) - root(root.right) if root else 0

def rotateRight(p):

    c = p.left
    p.left = c.right
    c.right = p

    p.height = 1 + max(getHeight(p.left) , getHeight(p.right))
    c.height = 1 + max(getHeight(c.left) , getHeight(c.right))

    return c 

def rotateLeft(p):

    c = p.right
    p.right = c.left
    c.left = p

    p.height = 1 + max(getHeight(p.left) , getHeight(p.right))
    c.height = 1 + max(getHeight(c.left) , getHeight(c.right))

    return c 

def insert(root,data):
    if root is None :
        return Node(data)   
    if data < root.data :
        root.left = insert(root.left,data)
    elif data > root.data :
        root.right = insert(root.right, data)
    else : return root

    root.height = 1 + max(getHeight(root.left),getHeight(root.right))

    balance = getBalance(root)

    if balance > 1 and data > root.left.data :
        root.left = rotateLeft(root.left)
        return rotateRight(root)

    if balance > 1 and data < root.left.data:
        return rotateRight(root)

    if balance < -1 and data > root.right.data :
        return rotateLeft(root)

    if balance < -1 and data < root.left.data :

        root.right = rotateRight(root.right)
        return rotateLeft(root)
    
    return root

def delete(root,data):
    if root is None : return None

    if data > root.data :
        root.right = delete(root.right,data)
    elif data < root.data :
        root.left = delete(root.left,data)
    else :
        if root.right is None and root.left is None :
            return None
        elif root.right is None :
            return root.left
        elif root.left is None :
            return root.right
        else :
            successor = root.right
            while successor.left :
                successor = root.left
            
            root.data = successor.data

            root.right = delete(root.right,successor.data)
    
    return root

