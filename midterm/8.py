class Node :
    def __init__(self,data,next):
        self.data = data
        self.next = next

class StackType :
    def __init__(self):
        self.top = None

    def isEmpty(self):
        return self.top == None
    
    def push(self,data):
        node  = Node(data,self.top)
        self.top = node

    def pop(self):
        if not self.isEmpty():
            p = self.top
            self.top = p.next
            return p.data
        else : pass

    def peek(self):
        if not self.isEmpty():
            return self.top.data
        
    def printList(self):
        p = self.top
        while p!= None :
            print("[%s] -> "%(p.data),end="")
            p = p.next
        print("\b\b\b\b   ")

    def sortedInsert(self,data):
        if (stack.isEmpty() or data > self.peek()):
            stack.push(data)
        else :
            temp = self.pop()
            self.sortedInsert(data)
            self.push(temp)


if __name__ == "__main__":
    stack = StackType()

    stack.push("A")


    list = ['R','E','D','F','C',"B"]

    for i,data in enumerate(list):
        stack.sortedInsert(data)

    stack.printList()

