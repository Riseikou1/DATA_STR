capacity = 6
size = 0
array = [None]*capacity

def isEmpty():
    return size == 0

def isFull():
    return size == capacity

def insert(pos,e):
    global size
    if not isFull() and 0<=pos<=size :
        for i in range(size,pos,-1):
            array[i] = array[i-1]
        array[pos] = e
        size += 1
    else :
        print("OverFlow or Invalid Position")

def delete(pos):
    global size
    if not isEmpty() and 0<=pos<size :
        e = array[pos]
        for i in range(pos,size-1):
            array[i] = array[i+1]
        size -= 1
        return e
    else :
        print("UnderFlow or Invalid Position")

def getEntry(pos):
    if 0<=pos<size :
        return array[pos]
    else :
        return None
    
if __name__ == "__main__":
    list = ['A','B','C','D']
    for ind,item in enumerate(list) :
        insert(ind,item)
    
    print(array[0:size])
    print(getEntry(2))