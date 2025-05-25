class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

class ListType:
    def __init__(self):
        self.tail = None
        self.size = 0

    def isEmpty(self):
        return self.tail is None

    def insertFirst(self, data):
        new_node = Node(data)
        if self.isEmpty():
            new_node.next = new_node
            self.tail = new_node
        else:
            new_node.next = self.tail.next
            self.tail.next = new_node
        self.size += 1

    def insertLast(self, data):
        new_node = Node(data)
        if self.isEmpty():
            new_node.next = new_node
        else:
            new_node.next = self.tail.next
            self.tail.next = new_node
        self.tail = new_node
        self.size += 1

    def deleteFirst(self):
        if self.isEmpty():
            print("List is empty. Nothing to delete.")
            return
        if self.size == 1:
            data = self.tail.data
            self.tail = None
        else:
            data = self.tail.next.data
            self.tail.next = self.tail.next.next
        self.size -= 1
        return data

    def deleteLast(self):
        if self.isEmpty():
            print("List is empty. Nothing to delete.")
            return
        if self.size == 1:
            data = self.tail.data
            self.tail = None
        else:
            current = self.tail.next
            while current.next != self.tail:
                current = current.next
            data = current.next.data
            current.next = self.tail.next
            self.tail = current
        self.size -= 1
        return data

    def insert(self, pos, data):
        if not 0 < pos <= self.size+1:
            print('pisda')
            return
        
        if pos == 1:
            self.insertFirst(data)
        elif pos == self.size+1:
            self.insertLast(data)
        else:
            p = self.tail.next
            for _ in range(1, pos-1):
                p = p.next
            node = Node(data)
            node.next = p.next
            p.next = node
            self.size += 1

    def delete(self, pos):
        if not 0 < pos <= self.size:
            print('pisda')
            return
        
        if pos == 1:
            self.deleteFirst()
        elif pos == self.size:
            self.deleteLast()
        else:
            p = self.tail.next
            for _ in range(1, pos-1):
                p = p.next
            data = p.next.data
            p.next = p.next.next
            self.size -= 1
            return data
            
    def printList(self):
        if self.isEmpty():
            print("The list is empty.")
            return

        p = self.tail.next
        for _ in range(self.size):
            print(f'[{p.data}]', end=" -> " if p.next != self.tail.next else "")
            p = p.next
        print()

        
if __name__ == "__main__":
    L = ListType()
    L.insertFirst('O')
    print("Tail node of the list : ",L.tail.data)
    L.insertLast('B')
    L.insertLast('C')
    print("Tail node of the list : ",L.tail.data)
    L.printList()
    L.insertLast('A')
    print("Tail node of the list : ",L.tail.data)
    L.printList()

    L.insertFirst('F')
    L.printList()
    print("Adding 'G' to at the first of the list.")
    L.insertFirst('G')
    L.printList()
    print("Tail node of the list : ",L.tail.data)
    print(f"Deleting the head node : {L.tail.next.data}")
    L.deleteFirst()
    L.printList()

    print(f"Deleting the last node : {L.tail.data}")
    L.deleteLast()
    L.printList()

    print(f"Deleting the last node : {L.tail.data}")
    L.deleteLast()
    L.printList()