class CircularQueue :
    def __init__(self,capacity =15
                 ):
        self.capacity = capacity
        self.front = 0
        self.rear = 0
        self.queue = [None]*self.capacity

    def isEmpty(self):
        return self.rear == self.front
    
    def isFull(self):
        return self.front == (self.rear+1) % self.capacity
    
    def enqueue(self,e) :
        if not self.isFull():
            self.rear = (self.rear+1) % self.capacity
            self.queue[self.rear] = e
        else :
            print("OverFlow!")
    
    def dequeue(self):
        if not self.isEmpty():
            self.front = (self.front+1) % self.capacity
            return self.queue[self.front]
        else :
            print("UnderFlow!")

    def peek(self):
        if not self.isEmpty():
            return self.queue[(self.front+1) % self.capacity]

    def display(self):
        print("Front : %d, Rear : %d "%(self.front,self.rear))

        i = self.front
        while i!= self.rear:
            i = (i+1)%self.capacity
            print('[%c]'% self.queue[i],end= " ")
        print()


if __name__ == "__main__":

    Q = CircularQueue()
    data = ['A','B','C','D','E']

    for e in data :
        Q.enqueue(e)

    Q.display()

    print('Dequeue -->',Q.dequeue())
    print('Dequeue -->',Q.dequeue())

    Q.display()

    Q.enqueue('F')
    Q.enqueue('G')
    Q.enqueue('H')
    Q.enqueue('I') # ene deer ali hediin rear ni 9 boltson baigaa.
    Q.display()
    Q.enqueue('J')
    Q.display()
    
    Q.enqueue('K')
    Q.enqueue('X')
    Q.display()

    for i , e in enumerate(Q.queue) :
        print(f"index : {i} , element : {e}")
    
