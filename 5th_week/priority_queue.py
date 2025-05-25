class PriorityQueue :
    def __init__(self,capacity = 10):
        self.capacity = capacity
        self.array = [None]*capacity
        self.size = 0

    def isFull(self):
        return self.size == self.capacity
    
    def isEmpty(self) :
        return self.size == 0

    def enqueue(self,e):
        if not self.isFull():
            self.array[self.size] = e
            self.size += 1

    def dequeue(self):
        highest = self.findMaxIndex()
        if highest != -1 :
            self.size -= 1
            self.array[highest],self.array[self.size] = \
                self.array[self.size] , self.array[highest]
            return self.array[self.size]
        
    def peek(self):
        highest = self.findMaxIndex()
        if highest != -1 :
            return self.array[highest]
        
    def findMaxIndex(self):
        max = 0
        for i in range(self.size) :
            if self.array[i] > self.array[max] :
                max = i
        return max
    
    def items(self):
        return (self.array[0:self.size])
    
q = PriorityQueue()
q.enqueue(34)
q.enqueue(18)
q.enqueue(27)
q.enqueue(15)
q.enqueue(45)

print("Queue :",q.items())

# while not q.isEmpty():
#     print("Max Priority = ",q.dequeue())

print("Dequeue-ing for 2 times.")
q.dequeue()
q.dequeue()
        
print("Queue:",q.items())

print("highest for now :",q.peek())