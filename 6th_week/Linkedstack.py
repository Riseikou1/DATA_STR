class Node :
    def __init__(self,data,next=None):
        self.data = data
        self.next = next

class StackType :
    def __init__(self):
        self.top = None
        self.size = 0

    def isEmpty(self):
        return self.size == 0
    
    def push(self,data):
        node = Node(data)
        node.next = self.top
        self.top = node
        self.size += 1 

    def printList(self):
        p = self.top

        while p != None :
            print('[| %s |]'%(p.data))
            print("-------")
            p = p.next
                
    def pop(self):
        if not self.isEmpty():
            p = self.top
            self.top = p.next
            self.size -= 1
            return p.data
        else : print("Error!")


if __name__ == "__main__":
    S = StackType()

    print("Pushing A to the stack")
    S.push('A')
    S.printList()
    print("Pushing 'B' to the stack.")
    S.push('B')
    S.printList()
    print(f"Head Data : {S.top.data}")

    print("Pushing 'C' to the stack.")
    S.push('C')
    S.printList()
    print(f"Deleting the head Node: {S.pop()}")
    S.printList()
