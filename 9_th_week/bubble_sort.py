def printStep(A,idx):
    print('     Step %d: '%idx, end="")
    print(A)


def bubbleSort(A):  # bubble sorts makes largest data at the end.
    n = len(A)

    for i in range(n-1):
        flag = False
        for j in range(1, n-i):
            if(A[j-1] > A[j]):
                A[j-1], A[j] = A[j], A[j-1]
                flag = True

        if not flag :
            break
    
        printStep(A,i+1)
    print()

if __name__ == "__main__":
    data = [5,3,8,4,9,1,6,2,7]

    L = list(data)
    print("Before: ",L)
    print()
    bubbleSort(L)
    print("Insertion: ",L)
