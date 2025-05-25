class ArrayList:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.array = [None] * capacity
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity

    def delete(self, pos):
        if not self.isEmpty() and 0 <= pos < self.size:
            e = self.array[pos]
            for i in range(pos, self.size - 1):
                self.array[i] = self.array[i + 1]
            self.array[self.size-1] = None  # ingehgui bol last iter ni duplicated nahui.
            self.size -= 1
            return e
        else:
            raise IndexError("Invalid position or empty list.")

    def insert(self, pos, e):
        if not self.isFull() and 0 <= pos <= self.size:
            for i in range(self.size, pos, -1):
                self.array[i] = self.array[i - 1]
            self.array[pos] = e
            self.size += 1
        else:
            raise IndexError("Invalid position or full list.")
        

    def getEntry(self, pos):
        if 0 <= pos < self.size:
            return self.array[pos]
        else:
            return None

    def __str__(self):
        return str(self.array[0:self.size])

    def contains(self, e):
        for i in range(self.size):
            if self.array[i] == e:
                return True
        return False
    
    def change(self,pos,data):
        if 0 <= pos < self.size :
            self.array[pos] = data


# Test code
if __name__ == "__main__":
    Array = ArrayList()

    Array.insert(0, 10)             # [10]
    Array.insert(1, 20)             # [10, 20]
    Array.insert(1, 30)             # [10, 30, 20]
    Array.insert(2, 40)             # [10, 30, 40, 20]
    Array.insert(Array.size, 50)    # [10, 30, 40, 20, 50]
    Array.insert(2, 60)             # [10, 30, 60, 40, 20, 50]

    print("ArrayList contents:", Array)

    # Delete test
    removed = Array.delete(3)       # removes element at index 3
    print("Removed:", removed)
    print("After deletion:", Array)

    # Contains test
    print("Contains 60?", Array.contains(60))
    print("Contains 999?", Array.contains(999))
