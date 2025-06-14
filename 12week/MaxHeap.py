# Max-Heap is a binary tree where the value of each parent node is greater than or equal to the values of its child nodes.

N = 20

class MaxHeap :
    def __init__(self):
        self.heap = [None]* N
        self.heapSize = 0

    def upHeap(self):
        i = self.heapSize  # last element.
        key = self.heap[i]
        while (i != 1) and key > self.heap[i // 2]:
            self.heap[i] = self.heap[i // 2]
            i = i // 2
        self.heap[i] = key


    def insertItem(self,item):
        self.heapSize += 1
        self.heap[self.heapSize] = item
        self.upHeap()

    def downHeap(self):
        key = self.heap[1]
        p = 1
        c = 2
        while c <= self.heapSize :
            if (c<self.heapSize) and (self.heap[c+1]>self.heap[c]):
                c += 1
            if key >= self.heap[c] :
                break
            self.heap[p] = self.heap[c]
            p = c
            c *= 2
        self.heap[p] = key

    def deleteItem(self):
        key = self.heap[1]
        self.heap[1] = self.heap[self.heapSize]
        self.heapSize -= 1
        self.downHeap()
        return key

if __name__ == "__main__":
    H = MaxHeap()
    data = [3,7,6,5,4,9,2,1,3]

    for d in data :
        H.insertItem(d)
        print("Heap :",H.heap[1:H.heapSize+1]) # first node is not being used for easier parent-child calculations.

    print("------------------------")
    H.insertItem(8)
    print("Inserting 8 to the shit.")
    print("Heap :",H.heap[1:H.heapSize+1])
    print("------------------------")

    print("[%d] is deleted"%H.deleteItem())
    print("Heap :",H.heap[1:H.heapSize+1])


