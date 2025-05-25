class SortedArraySet:
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.array = [None] * capacity
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity

    def __str__(self):
        return str(self.array[:self.size])

    def contains(self, e):
        for i in range(self.size):
            if self.array[i] == e:
                return True
        return False

    def insert(self, e):
        if self.contains(e) or self.isFull():
            return
        self.array[self.size] = e
        self.size += 1
        for i in range(self.size - 1, 0, -1):
            if self.array[i - 1] < self.array[i]:
                break
            self.array[i - 1], self.array[i] = self.array[i], self.array[i - 1]

    def delete(self, e):
        if not self.contains(e) or self.isEmpty():
            return None
        i = 0
        while self.array[i] < e:
            i += 1  # i is now the index to be deleted.
        self.size -= 1
        while i < self.size:  
            self.array[i] = self.array[i + 1]  # and then shift the elements to overwrite the element to be deleted.
            i += 1
        self.array[self.size] = None  # Clear leftover value

    def union(self, setB):
        setC = SortedArraySet()
        i = j = 0
        while i < self.size and j < setB.size:
            a = self.array[i]
            b = setB.array[j]
            if a == b:
                setC.insert(a)
                i += 1
                j += 1
            elif a < b:
                setC.insert(a)
                i += 1
            else:
                setC.insert(b)
                j += 1
        while i < self.size:
            setC.insert(self.array[i])
            i += 1
        while j < setB.size:
            setC.insert(setB.array[j])
            j += 1
        return setC


if __name__ == "__main__":
    import random 
    setA = SortedArraySet()
    setB = SortedArraySet()
    for i in range(5):
        setA.insert(random.randint(1,9))
        setB.insert(random.randint(5,11))

    print("SetA: ",setA)
    print("SetB: ",setB)

    setC = setA.union(setB)

    print("Union of setA and setB : ",setC)