class ArrayStack :
    def __init__(self,capacity=30):
        self.capacity = capacity
        self.top = -1
        self.array= [None]*capacity

    def isEmpty(self):
        return self.top == -1
    
    def isFull(self):
        return self.top+1 == self.capacity
    
    def push(self,e):
        if not self.isFull():
            self.top += 1
            self.array[self.top] = e 
    
    def pop(self):
        if not self.isEmpty() :
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
            print("-----")
        print()

def evalPostfix(expr) :
    s = ArrayStack(100)
    
    for token in expr :
        if token in "+-*/" :
            val2 = s.pop()
            val1 = s.pop()
            if   (token == '+') : s.push(val1+val2)
            elif (token == '-') : s.push(val1-val2)
            elif (token == '*') : s.push(val1*val2)
            elif (token == '/') : s.push(val1/val2)

        else :
            s.push(float(token))

    return s.pop()
def predence(pisda) :
    if pisda == "(" or pisda == ")" : return 0
    elif pisda == "+" or pisda == "-" : return 1
    elif pisda == "*" or pisda == "/" : return 2
    else : return -1

def Infix2Postfix(expr):
    s = ArrayStack(100)
    output = list()
    for term in expr:
        if term == "(":
            s.push(term)
        elif term == ")":
            while not s.isEmpty():
                el = s.pop()
                if el == "(":
                    break
                output.append(el)
        elif term in "+-/*":
            while not s.isEmpty():
                op = s.peek()
                if predence(op) >= predence(term):
                    output.append(s.pop())
                else:
                    break
            s.push(term)
        else:
            output.append(term)
    while not s.isEmpty():
        output.append(s.pop())
    return output


infix1 = ['8','/','2','-','3','+','(','3','*','2',')']
infix2 = ['1','/','2','*','4','*','(','1','/','4',')']

postfix1 = Infix2Postfix(infix1)
postfix2 = Infix2Postfix(infix2)

result1 = evalPostfix(postfix1)
result2 = evalPostfix(postfix2)

print("expression: ",infix1)
print("calculation ",postfix1)
print("result :",result1)

print("expression: ",infix2)
print("calculation ",postfix2)
print("result :",result2)
