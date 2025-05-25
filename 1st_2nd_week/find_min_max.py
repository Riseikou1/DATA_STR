def find_min_max(A):
    max = A[0]
    min =A[0]
    for i in range(1,len(A)):
        if max < A[i]: max = A[i]
        if min > A[i]: min = A[i]
    return max,min


data = [5,4,3,2,5,6,9]
max , min = find_min_max(data)
print("(max,min) =",(max,min))
