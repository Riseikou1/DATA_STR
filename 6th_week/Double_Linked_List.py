class Node :
    def __init__(self,data,prev=None,next=None):
        self.data = data
        self.prev = prev
        self.next = next

class DListType :
    def __init__(self):
        self.size = 0
        self.head = None
        self.tail = None

    def isEmpty(self):
        return self.size == 0

    def insertFirst(self, data):
        node = Node(data, next=self.head)
        if self.isEmpty():
            self.head = self.tail = node
        else:
            self.head.prev = node
            self.head = node
        self.size += 1
        
    def insertLast(self, data):
        node = Node(data, prev=self.tail)
        if self.isEmpty():
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self.size += 1


    def deleteFirst(self):
        if not self.isEmpty():
            data = self.head.data

            self.head = self.head.next
            if self.head == None :
                self.tail = None
            else :
                self.head.prev = None
            self.size -= 1
            return data

    def deleteLast(self):
        if not self.isEmpty():
            data = self.tail.data

            self.tail = self.tail.prev
            if self.tail == None :
                self.head = None
            else :
                self.tail.next = None
            self.size -= 1
            return data
        
    def delete(self, pos):
        if not 0 < pos <= self.size:
            return
        if not self.isEmpty():
            if pos == 1:
                return self.deleteFirst()
            elif pos == self.size:
                return self.deleteLast()
            else:
                p = self.head
                for _ in range(1, pos-1):
                    p = p.next
                target = p.next
                p.next = target.next
                if target.next:  # checking this. cuz we might delete node before tail. that shit's next.next will be None so we needed to this line.
                    target.next.prev = p
                self.size -= 1
                return target.data

    def insert(self, data, pos):
        if not 0 < pos <= self.size + 1:
            print("Invalid Position")
            return
        if self.isEmpty() or pos == 1:
            return self.insertFirst(data)
        elif pos == self.size + 1:
            return self.insertLast(data)
        else:
            p = self.head
            for _ in range(1, pos - 1):
                p = p.next
            node = Node(data,p.next)
            node.prev = p
            p.next.prev = node
            p.next = node
            self.size += 1


    def printList(self):
        p = self.head
        while p!= None :
            print('[%s] <-> '%p.data , end='')
            p = p.next
        print("\b\b\b\b      ")


if __name__ == "__main__":
    DL = DListType()

    DL.insertFirst('A')
    DL.insertFirst('B')
    DL.insertFirst('C')
    DL.insertFirst('D')
    DL.printList()

    DL.insertLast('T')
    DL.insertLast("E")
    DL.printList()

    print('Head element : [%s] is deleted.'%(DL.deleteFirst()))
    print('Head element : [%s] is deleted.'%(DL.deleteFirst()))
    DL.printList()

    print('Tail element : [%s] is deleted.'%(DL.deleteLast()))
    print('Tail element : [%s] is deleted.'%(DL.deleteLast()))
    DL.printList()


    listLetter = ['F','U','O','P','C','V']
    

    for data in (listLetter):
        DL.insertLast(data)

    print('Head element : [%s] is deleted.'%(DL.deleteFirst()))
    print('Tail element : [%s] is deleted.'%(DL.deleteLast()))

    DL.printList()
    print(DL.tail.prev.data)
    print(DL.tail.data)
    print(DL.head.data)
    print(DL.head.next.data)
    print(DL.head.next.next.next.prev.data)


