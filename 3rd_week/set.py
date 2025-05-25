class ArraySet:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.array = [None] * capacity
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity

    def getEntry(self, pos):
        if 0 <= pos < self.size:
            return self.array[pos]
        else:
            return None

    def __str__(self):
        #return "{" + ", ".join(str(self.array[i]) for i in range(self.size)) + "}"
        return str(self.array[0:self.size])

    def contains(self, e):
        for i in range(self.size):
            if self.array[i] == e:
                return True
        return False

    def insert(self, e):
        if not self.contains(e) and not self.isFull():
            self.array[self.size] = e
            self.size += 1

    def delete(self, e):
        for i in range(self.size):
            if self.array[i] == e:
                self.array[i] = self.array[self.size - 1]
                self.size -= 1
                return

    def union(self, setB):
        setC = ArraySet()
        for i in range(self.size):
            setC.insert(self.array[i])
        for i in range(setB.size):
            if not setC.contains(setB.array[i]):
                setC.insert(setB.array[i])
        return setC

    def intersect(self, setB):
        setC = ArraySet()
        for i in range(self.size):
            if setB.contains(self.array[i]):
                setC.insert(self.array[i])
        return setC

    def difference(self, setB):
        setC = ArraySet()
        for i in range(self.size):
            if not setB.contains(self.array[i]):
                setC.insert(self.array[i])
        return setC
    
    def Symmetric_difference(self, setB):
        setC = ArraySet()  # New set to store the symmetric difference
        # Insert items from self that are not in setB
        for i in range(self.size):
            if not setB.contains(self.array[i]):
                setC.insert(self.array[i])

        # Insert items from setB that are not in self
        for i in range(setB.size):
            if not self.contains(setB.array[i]):
                setC.insert(setB.array[i])

        return setC  # Return the symmetric difference set

    
    """def difference(self,setB):
        for i in range(self.size):
            if self.contains(setB.array[i]):
                self.delete(self.array[i])
        return self"""


if __name__ == "__main__":
    Array = ArraySet()

    Array.insert(10)
    Array.insert(20)
    Array.insert(30)
    Array.insert(40)
    Array.insert(50)

    print("Set A:",Array)

    B = ArraySet()

    B.insert(60)
    B.insert(70)
    B.insert(80)
    B.insert(90)
    B.insert(100)
    B.insert(10)
    B.insert(20)
    print("Set B:",B)

    C = Array.union(B)
    print("UNION:", C)

    D = Array.intersect(B)
    print("INTERSECT:", D)

    F = Array.difference(B)
    print("DIFFERENCE (A-B):", F)

    temuujin = Array.Symmetric_difference(B)
    print("SYMMETRIC DIFFERENCE (A^B):", temuujin)