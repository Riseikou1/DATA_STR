class Node :
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def getHeight(root):
    if root is None :
        return 0
    return 1 + max(getHeight(root.left),getHeight(root.right))

def getBalance(root):
    return getHeight(root.left) - getHeight(root.right)


def rotateRight(p):
    c = p.left
    p.left = c.right
    c.right = p

    return c

def rotateLeft(p):
    c = p.right
    p.right = c.left
    c.left = p

    return c

def balancer(root,data):
    balance = getBalance(root)

    if balance > 1 :
        if data > root.left.data :
            root.left = rotateLeft(root.left)
        return rotateRight(root)
    
    if balance < -1 :
        if data < root.right.data:
            root.right = rotateRight(root.right)
        return rotateLeft(root)
    
    return root

def insert(root,data):
    if root is None :
        return Node(data)
    if data < root.data :
        root.left = insert(root.left,data)
    elif data > root.data :
        root.right = insert(root.right,data)
    else :
        return root
    
    return balancer(root,data)


def delete(root,data):
    if root is None : 
        return None
    
    if data < root.data :
        root.left = delete(root.left,data)
    elif data > root.data :
        root.right = delete(root.right,data)
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
                successor = successor.left
            root.data = successor.data 
            root.right = delete(root.right,successor.data)
    
    return balancer(root,data)

def preOrder(root):
    if root is not None :
        print(root.data, end=" ")
        preOrder(root.left)
        preOrder(root.right)

def display(root,msg):
    print(msg, end=" ")
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
