class Node :
    def __init__(self,data,next=None,prev=None):
        self.data = data
        self.next = next
        self.prev = prev

class DListType :
    def __init__(self):
        self.size = 0
        self.head = Node(None)
        self.tail = Node(None)
        self.tail.prev = self.head
        self.tail.next = self.tail

        self.tail.next = self.head
        self.head.prev = self.tail

    def isEmpty(self):
        return self.size == 0
    
    def insertFirst(self,data):
        node = Node(data)
        if self.isEmpty():
            self.tail.prev = node
            node.next = self.tail
        else :
            node.next = self.head.next
            self.head.next.prev = node 
        node.prev = self.head
        self.head.next = node
        self.size += 1

    def insertLast(self,data):
        node = Node(data)
        if self.isEmpty():
            self.head.next = node
            node.prev = self.head
        else :
            self.tail.prev.next = node
            node.prev = self.tail.prev
        node.next= self.tail
        self.tail.prev = node
        self.size += 1

    def deleteFirst(self):
        if self.isEmpty():
            print("List is Empty")
            return
        else :
            data = self.head.next.data
            if self.size == 1 :
                self.tail.prev = self.head
                self.head.next = self.tail
            else :
                self.head.next.next.prev = self.head
                self.head.next = self.head.next.next 
            self.size -= 1
            return data

    def deleteLast(self):
        if self.isEmpty():
            print("List is Empty")
            return
        else :
            data = self.tail.prev.data
            if self.size == 1 :
                self.tail.prev = self.head
                self.head.next = self.tail
            else :
                self.tail.prev.prev.next = self.tail
                self.tail.prev = self.tail.prev.prev
            self.size -= 1
            return data

    def insert(self,data,pos):
        if not 0 < pos <= self.size + 1:
            print("Invalid Position")
            return
        if pos == 1 or self.isEmpty():
            return self.insertFirst(data)
        elif pos == self.size + 1:
            return self.insertLast(data)
        else :
            p = self.head
            for _ in range(pos-1):
                p = p.next
            node = Node(data,p.next)
            node.prev = p
            p.next.prev = node
            p.next = node
            self.size += 1

    def delete(self,pos):
        if not 0 < pos <= self.size:
            print("Invalid Position")
            return
        if pos == 1 :
            return self.deleteFirst()
        elif pos == self.size:
            return self.deleteLast()
        else :
            p = self.head
            for _ in range(pos):
                p = p.next
            data = p.data
            p.prev.next = p.next
            p.next.prev= p.prev
            self.size -= 1
            return data
        
    def printList(self):
        if self.isEmpty():
            print("Empty List")
            return

        p = self.head.next
        while p != self.tail:
            print(p.data, end=" <-> ")
            p = p.next
        print("\b\b\b\b    ")
        
    def reverse(self):
        if self.isEmpty():
            print("List is empty. Nothing to reverse.")
            return

        first = self.head.next
        last = self.tail.prev

        current = first

        while current != self.tail:
            current.prev, current.next = current.next, current.prev
            current = current.prev  # because next and prev swapped

        self.head.next = last
        last.prev = self.head

        self.tail.prev = first
        first.next = self.tail


if __name__ == "__main__":
    DL = DListType()

    DL.insert('A',1)
    DL.insert('B',2)
    DL.insert('C',3)
    DL.insert('D',4)
    DL.printList()

    print('1st node: [%s] is deleted.'%(DL.delete(1)))
    print('3rd node: [%s] is deleted.'%(DL.delete(3)))
    DL.printList()

    print("Adding so many bullshits.")
    listLetter = ['F','U','O','P','C','V']
    for i,data in enumerate(listLetter):
        DL.insert(data,i+1)
    DL.printList()

    print('2nd node: [%s] is deleted.'%(DL.delete(2)))
    print('7th node: [%s] is deleted.'%(DL.delete(7)))
    print('1st node: [%s] is deleted.'%(DL.delete(1)))
    DL.printList()

    print(DL.head.data)  # None butsaaj baigaag harj baigaa baih.


    DL.printList()
    DL.reverse()
    print("After reversing:")
    DL.printList()

