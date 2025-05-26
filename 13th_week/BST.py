class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

def insert(root, key):
    if root is None:
        return BSTNode(key)
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
    else:
        print("Can't insert the same number twice.")
    return root


def delete(root, key):
    if root is None:
        return None
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:  # Key to be deleted is found
        if root.left is None and root.right is None:  # No children
            return None
        elif root.left is None:  # Only right child
            return root.right
        elif root.right is None:  # Only left child
            return root.left
        else:  # Two children
            # Find the inorder successor (smallest in the right subtree)
            successor = root.right
            #sucessor = getminNode(root.right)
            while successor.left:
                successor = successor.left
            root.key = successor.key  # Replace root's key with successor's key
            root.right = delete(root.right, successor.key)  # Delete successor
    return root

def getminNode(root):
    while root != None and root.left != None:
        root = root.left
    return root

def preOrder(root):
    if root is not None:
        print("%2d" % root.key, end=" ")
        preOrder(root.left)
        preOrder(root.right)

def display(root, msg):
    print(msg, end='')
    preOrder(root)
    print()

if __name__ == "__main__":
    root = None
    data = [35, 18, 7, 26, 3, 22, 30, 12, 26, 68, 99]
    for key in data:
        root = insert(root, key)
        display(root, '[Insert %2d] : ' % key)
    print()

    root = delete(root,35)
    display(root, '[Delete %2d] : ' % 35)
    print(root.key)

