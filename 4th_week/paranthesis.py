from stack import Arraystack

def checkBrackets(str):
    S = Arraystack()

    for ch in str :
        if ch == '[' or ch == '{' or ch == '(' :
            S.push(ch)
        elif ch== ']' or ch == '}' or ch == ')':
            if S.isEmpty() :
                return False     # Error 2.
            else :
                open = S.pop()
                if  (ch==']' and open!='[') or \
                    (ch=='}' and open!='{') or \
                    (ch==')' and open!='(') :
                        return False   # Error 3.
                
    return S.isEmpty()  # bugd pop-rogdood , hoorondoo taarj duussanii daraa, stack empty baival True
                        # Ugui bol niit haaltnii too taaraagui tul # Error 1.

if __name__ == "__main__":
    s1 = '{ [ ( ) } ]'
    s2 = '( () ()'
    s3 = '( [ ] )'

    print(s1 , "-----> ",checkBrackets(s1))
    print(s2 , "-----> ",checkBrackets(s2))
    print(s3 , "-----> ",checkBrackets(s3))

     