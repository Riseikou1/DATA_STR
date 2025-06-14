import queue

class Node :
    def __init__(self,data,left=None,right=None):
        self.data = data
        self.left = left
        self.right = right

class binaryTRee:
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

    def postOrder(self,root):
        if root != None :
            self.postOrder(root.left)
            self.postOrder(root.right)
            print('[%c]' %root.data,end="") 

    def inOrder(self,root):
        if root != None :
            self.inOrder(root.left)
            print('[%c]' %root.data,end="") 
            self.inOrder(root.right)

    def preOrder(self,root):
        if root != None :
            print('[%c]' %root.data,end="") 
            self.preOrder(root.left)
            self.preOrder(root.right)

    def levelOrder(self,root):
        Q = queue.Queue()  # FIFO.
        Q.put(root)
        while not Q.empty():
            root = Q.get()
            print("[%c]"% root.data , end="")
            if root.left != None :
                Q.put(root.left)
            if root.right != None :
                Q.put(root.right)

        print()

    def nodeCount(self,root):
        if root == None :
            return 0
        else : 
            return 1 + self.nodeCount(root.left) + self.nodeCount(root.right)

    def isExternal(self,root):
        return root.left == None and root.right == None

    def leafCount(self, root):
        if root is None:            return 0
        if self.isExternal(root):   return 1
        return self.leafCount(root.left) + self.leafCount(root.right)

    def getHeight(self,root):
        if root is None  :      return 0
        else :
            return 1 + max(self.getHeight(root.left),self.getHeight(root.right))
        
    def treeReverse(self,root):
        if root != None :
            root.left, root.right = root.right, root.left 
            self.treeReverse(root.left)
            self.treeReverse(root.right)

if __name__ == "__main__":
    T = binaryTRee()
    N6 = Node('F')
    N5 = Node('E')
    N4 = Node('D')
    N3 = Node('C',N6,None)
    N2 = Node('B',N4,N5)
    N1 = Node('A',N2,N3)
    print("Pre  : ",end="") ; T.preOrder(N1)  ; print()
    print("In   : ",end="") ; T.inOrder(N1)   ; print()
    print("Post : ",end="") ; T.postOrder(N1) ; print()
    print("lvl  : ",end="") ; T.levelOrder(N1); print()
    print(f"There is {T.nodeCount(N1)} node(s) in our tree.")
    print(f"There is {T.leafCount(N1)} leaves in our tree.")
    print(f"Our tree's height : {T.getHeight(N1)}")

    T.treeReverse(N1)
    print("Post traversing through Tree after reversing it.")
    T.postOrder(N1); print()

    T.insert(N1,'G')
    T.preOrder(N1); print()

"""                         'A'
                 'C'                  'B'
            None      'F'        'E'       'D'
    
"""