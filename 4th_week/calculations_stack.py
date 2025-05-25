from stack import Arraystack
from file_paranthesis import evalPostfix

def Infix2Postfix(expr):
    s = Arraystack(100)
    output = list()

    for term in expr:
        if term == "(":
            s.push(term)
        elif term == ")":
            while not s.isEmpty():
                op = s.pop()
                if op == "(":
                    break
                output.append(op)

        elif term in "+-*/":
            while not s.isEmpty() and precedence(s.peek()) >= precedence(term):
                output.append(s.pop())
            s.push(term)

        else:
            output.append(term)

    while not s.isEmpty():
        output.append(s.pop())

    return output


def precedence(op):
    if   op =="("  or op==')' : return 0
    elif op == "+" or op=='-' : return 1
    elif op == "*" or op=="/" : return 2
    else : return -1


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

