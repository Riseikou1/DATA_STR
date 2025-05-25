class Node :
    def __init__(self,data,next=None):
        self.data = data
        self.next = next

class ListType :
    def __init__(self):
        self.head = None
        self.size = 0

    def isEmpty(self):
        return self.size == 0
    
    def insertFirst(self,data):
        node = Node(data)
        node.next = self.head
        self.head = node
        self.size += 1 

    def insertLast(self, data):
        if self.isEmpty():
            self.insertFirst(data)
            return
        p = self.head
        while p.next:
            p = p.next
        p.next = Node(data)
        self.size += 1

    def deleteFirst(self):
        if not self.isEmpty():
            p = self.head
            self.head = p.next
            self.size -= 1
            return p.data
        else : print("Error!")

    def deleteLast(self):
        if not self.isEmpty():
            if self.size == 1:
                data = self.head.data
                self.head = None
            else:
                p = self.head
                while p.next.next:
                    p = p.next
                data = p.next.data
                p.next = None
            self.size -= 1
            return data

    def insert(self, pos, data):
        if pos <= 0 or pos > self.size + 1:
            return  # Invalid position

        dummy = Node(None, self.head)
        p = dummy
        for _ in range(pos - 1):
            p = p.next

        p.next = Node(data, p.next)

        self.head = dummy.next
        self.size += 1
    
    def delete(self, pos):
        if self.isEmpty() or pos <= 0 or pos > self.size:
            return  # Invalid position or empty list

        dummy = Node(None, self.head)
        p = dummy
        for _ in range(pos - 1):
            p = p.next

        data = p.next.data
        p.next = p.next.next

        self.head = dummy.next
        self.size -= 1
        return data

    def printList(self):
        p = self.head

        while p != None :
            print('[%s] -> '%(p.data),end='')
            p = p.next
        print('\b\b\b    ')


if __name__ == "__main__":
    L = ListType()
    print(L.isEmpty())

    L.insertFirst('A')
    L.printList()
    L.insertFirst('B')
    L.printList()
    print(L.head.data)

    L.insert('C',1)
    L.insert('D',4)
    L.insert('E',3)

    print(L.isEmpty())
    L.printList()
    print(L.head.data)
    print(f"Deleting the head Node : {L.deleteFirst()}")
    L.printList()


    
    