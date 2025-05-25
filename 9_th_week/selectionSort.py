import time 

start =time.time()

def printStep(A,idx):
    print("   Step %d : " %idx,end="")
    print(A)

def selectionSort(A):  # hamgiin baga toog ni ehleed urd ni avchraad shaah zamaar ajillana.
    n = len(A)
    for i in range(n-1):
        minIdx = i
        for j in range(i+1,n):
            if A[j] < A[minIdx]:
                minIdx = j
        if minIdx != i :
            A[i],A[minIdx] = A[minIdx] , A[i]
            printStep(A,i+1)
    


if __name__ == "__main__":
    data = [5,3,8,4,9,1,6,2,7]

    print("Before sorting the array: ",data)

    selectionSort(data)

    print("After sorting the array:  ",data)

    end = time.time()

    print(f"Time used : {(end-start):.8f} seconds", )