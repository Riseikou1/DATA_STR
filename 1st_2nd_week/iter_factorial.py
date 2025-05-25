def factorial_iter(n):
    result = 1
    for k in range(1,n+1):
        result *= k
    return result

t = factorial_iter(5)
print(t)