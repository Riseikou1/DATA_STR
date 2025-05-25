M = 13
class Node:
    def __init__(self,data,next=None):
        self.data = data
        self.next = next

class HashTable :
    def __init__(self):
        self.table = [None] * M

    def hashFn(self,key):
        return key % M
    
    def insert(self,key):
        bucket = self.hashFn(key)

        node = Node(key)
        node.next = self.table[bucket]
        self.table[bucket] = node

    def display(self):
        for i in range(M):
            print("HT[%2d] : "%(i),end="")
            n = self.table[i]

            while n is not None :
                print(n.data , end=" ")
                n = n.next
            print()

if __name__ == "__main__":
    HT = HashTable()
    import random
    for _ in range(20):
        HT.insert(random.randint(1, 100))
    HT.display()
