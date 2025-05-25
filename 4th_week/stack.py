class Arraystack :
    def __init__(self,capacity=30):
        self.capacity = capacity
        self.top = -1
        self.array = [None]*capacity

    def isEmpty(self):
        return self.top == -1

    def isFull(self):
        return self.top == self.capacity -1

    def push(self,e):
        if not self.isFull():
            self.top +=1
            self.array[self.top] = e

    def pop(self):
        if not self.isEmpty():
            e = self.array[self.top]
            self.top -= 1
            return e
        else :
            print("UnderFlow")
            return None


    def peek(self):
        if not self.isEmpty():
            return self.array[self.top]


    def print(self):
        for i in range(self.top,-1,-1):
            print(" |%d| " % (self.array[i]))
            print('-----')
        print()

        
if __name__ == "__main__":

    stack = Arraystack(20)

    data = [5,3,8,1,2,7]

    for e in data :
        stack.push(e)
    
    peek = stack.peek()
    print(peek)

    stack.print()
    print()

    print("last element was popped: " , stack.pop(),"\n")

    stack.print()

