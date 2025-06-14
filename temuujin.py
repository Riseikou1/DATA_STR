import queue

class Node :
    def __init__(self,data,left=None,right=None):
        self.data = data
        self.right = right
        self.left = left

class BinaryTree :
    def __init__(self):
        self.root = None

    def insert(self,root,data):
        if root is None :
            return Node(data)
        
        if root.data > data :
            root.left = self.insert(root.left,data)

        elif root.data < data :
            root.right = self.insert(root.right,data)

        return root
        
    def getHeight(self,root):   

        if root is None : return 0
        else :
            return 1 + max(self.getHeight(root.left), self.getHeight(root.right))
    
    def leafCount(self,root):

        if root is None : return None
        if root.left is None and root.right is None : return 1

        return self.leafCount(root.left) + self.leafCount(root.right)


    def nodeCount(self,root):
        if root is None :
            return 0
        else :
            return 1 + self.nodeCount(root.left) + self.nodeCount(root.right)


    
    def inOrder(self,root):
        if root is not None :
            self.inOrder(root.left)
            print(root.data , end= " ")
            self.inOrder(root.right)

    def postOrder(self,root):
        if root is not None :
            self.postOrder(root.left)
            self.postOrder(root.right)
            print(root.data, end=" ")

    def preOrder(self,root):
        if root is not None :
            print(root.data, end= " ")
            self.preOrder(root.left)
            self.preOrder(root.right)

    def levelOrder(self,root):
        q = queue.Queue()
        q.put(root)
        while not q.empty():
            root = q.get()
            print(root.data , end=" ")
            if root.left is not None :
                q.put(root.left)
            if root.right is not None :
                q.put(root.right)
        
            print()
    
    def treeReverse(self,root):
        if root is not None :
            root.right, root.left = root.left , root.right
            self.treeReverse(root.left)
            self.treeReverse(root.right)
        