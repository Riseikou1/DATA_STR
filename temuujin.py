class Node :
    def __init__(self,data,next=None,prev=None):
        self.data = data
        self.next = next
        self.prev = prev

class Listtype:
    def __init__(self):
        self.size = 0
        self.head = Node(None)
        self.tail = Node(None)
        self.tail.prev = self.head
        self.head.next = self.tail
        self.tail.next = self.head
        self.head.prev = self.tail

    def isEmpty(self):
        return self.size == 0
    
    def insertFirst(self,data):
        node = Node(data,prev=self.head)
        if self.isEmpty():
            node.next = self.tail
            self.tail.prev = node
        else :
            self.head.next.prev = node
            node.next = self.head.next
        self.head.next = node
        self.size += 1
        

    def insertLast(self,data):
        node = Node(data,next=self.tail)
        if self.isEmpty():
            self.head.next = node
            node.prev = self.head
        else :
            self.tail.prev.next = node
            node.prev = self.tail.prev
        self.tail.prev = node
        self.size += 1
        
    def deleteFirst(self):
        if not self.isEmpty():
            data = self.head.next
            if self.size == 1:
                self.tail.prev = self.head
                self.head.next = self.tail
            else :
                self.head.next.next.prev = self.head
                self.head.next = self.head.next.next
            self.size -= 1
            return data
    
    def deleteLast(self):
        if not self.isEmpty():
            data = self.tail.prev
            if self.size == 1:
                self.tail.prev = self.head
                self.head.next = self.tail
            else :
                self.tail.prev.prev.next = self.tail
                self.tail.prev = self.tail.prev.prev
            self.size -= 1
            return data  
        
    def insert(self,pos,data):
        if not 0 < pos <= self.size + 1:
            print("Invalid position.")
            return 
        
        if pos == 1:
            return self.insertFirst(data)
        elif pos == self.size + 1:
            return self.insertLast(data)
        else :
            p = self.head.next
            for _ in range(1,pos-1):
                p = p.next
            
            node = Node(data,prev=p,next=p.next)
            p.next.prev = node
            p.next = node
            self.size += 1

    def delete(self,pos):        
        if not 0 < pos <= self.size:
            print("Invalid position.")
            return 
        
        if pos == 1:
            return self.deleteFirst()
        elif pos == self.size :
            return self.deleteLast()
        else :
            p = self.head
            for _ in range(0,pos-1):
                p = p.next

            data = p.next.data
            p.next.next.prev = p
            p.next = p.next.next
            self.size -= 1

            return data
        