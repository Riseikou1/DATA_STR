import random

def insertionSort(A):
    n = len(A)
    for i in range(1,n):
        key = A[i]
        j = i -1 
        while j >=0 and A[j] > key :
            A[j+1] = A[j]
            j -= 1
        A[j+1] = key


def seqSearch(list,key):

    n = len(list)
    for i in range(n):
        if list[i] == key:
            return i
        else :  
            return -1
        
def iBinarySearch(A,key):
    low = 0
    high = len(A)-1

    while(low <= high):
        middle = (high +low)// 2
        if A[middle]==key :
            return middle
        
        elif A[middle] < key :  # hervee dundiin value ih baival, suuliin hagasiig shaana.
            low = middle + 1
        else :
            high = middle - 1 

    return -1

def rBinarySearch(A,key,low,high):    # recursive binary search

    if low <= high:
        mid = (low+high)//2
        print(A[mid],end='  ')

        if key == A[mid]:
            return mid
        elif key < A[mid] :
            return rBinarySearch(A,key,low,mid-1)
        else :
            return rBinarySearch(A,key,mid + 1,high)
        
    return -1
    

if __name__ == "__main__":
    A = []
    for i in range(15):
        A.append(random.randint(1,100))

    insertionSort(A)
    print('A[] = ',A)

    key = int(input("Input Search Key : "))

    #idx = seqSearch(A,key)
    #idx = iBinarySearch(A,key)
    idx = rBinarySearch(A,key,0,len(A)-1)

    if idx != -1 :
        print("Key found at position : ",idx+1)
    else : print("Key wasnt' found")