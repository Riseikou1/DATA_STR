from stack import Arraystack
S = Arraystack()

msg = input("String Input: ")
for c in msg : 
    S.push(c)

print("Print String: ",end="")
while not S.isEmpty():
    print(S.pop(),end="")

print()


temuujin = Arraystack(10)
for i in range(1,6):
    temuujin.push(i)
print(temuujin.array)
