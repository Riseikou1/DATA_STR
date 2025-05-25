class ArrayQueue :
    def __init__(self,capacity =100):
        self.capacity = capacity
        self.front = -1
        self.rear = -1
        self.queue = [None]*self.capacity

    def isEmpty(self):
        return self.rear == self.front
    
    def isFull(self):
        return self.rear == self.capacity - 1
    
    def enqueue(self,e) :
        if not self.isFull():
            self.rear += 1
            self.queue[self.rear] = e
        else :
            print("OverFlow!")
    
    def dequeue(self):
        if not self.isEmpty():
            self.front += 1
            return self.queue[self.front]
        else :
            print("UnderFlow!")

    def display(self):
        print("Front : %d, Rear : %d "%(self.front,self.rear))

        print(self.queue[self.front+1:self.rear+1])


if __name__ == "__main__":
    Q = ArrayQueue()
    data = ['A','B','C','D','E']

    for e in data :
        Q.enqueue(e)


    Q.display()

    print('Dequeue -->',Q.dequeue())
    print('Dequeue -->',Q.dequeue())

    Q.display()

    Q.enqueue('F')
    Q.display()


    
