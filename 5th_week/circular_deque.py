from circular_queue import CircularQueue

class CircularDeque(CircularQueue):
    def __init__(self):
        super().__init__()

    def addRear(self, item): self.enqueue(item)
    def deleteFront(self): return self.dequeue()
    def getFront(self):return self.peek()

    def getRear(self):
        if not self.isEmpty():
            return self.queue[self.rear]
        else : print("Error.")

    def addFront(self, item):
        if not self.isFull():
            self.queue[self.front] = item
            self.front = (self.front - 1 + self.capacity) % self.capacity
        else: pass

    def deleteRear(self):
        if not self.isEmpty():
            item = self.queue[self.rear]
            self.rear = (self.rear - 1 + self.capacity) % self.capacity
            return item
        else: pass

    def peekFront(self):
        if not self.isEmpty():
            return self.deque[(self.front + 1) % self.capacity]

    def peekRear(self):
        if not self.isEmpty():
            return self.deque[self.rear]

    def __str__(self):
        if self.front <= self.rear:
            return str(self.queue[self.front+1:self.rear + 1])
        else:
            return str(self.queue[self.front+1:] + self.queue[:self.rear + 1])

dq = CircularDeque()

# Add even numbers to the rear and odd numbers to the front
for i in range(9):
    if i % 2 == 0:
        dq.addRear(i)
    else:
        dq.addFront(i)

print("Even numbers are in the rear, odd ones in the front")
print("Using str : " ,dq)
print("Printing underhood real deque : ", dq.queue)

for i in range(2):
    dq.deleteFront()

for i in range(3):
    dq.deleteRear()

print("delete front*2 , rear*3 times: ", dq)
print("Printing underhood real deque : ", dq.queue)

print("Adding some shits in the Front")
for i in range(9, 14):
    dq.addFront(i)

print("Using str : ",dq)

print("Printing underhood real deque : ", dq.queue)

print("Getting front element : ",dq.getFront())
print("Getting last element : ",dq.getRear())