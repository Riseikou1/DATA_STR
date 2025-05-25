def factorial(n):
    if n==1 : return 1
    else : return n*factorial(n-1)

t = factorial(5)
print(t)