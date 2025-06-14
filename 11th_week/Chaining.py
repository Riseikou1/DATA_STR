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

    def delete(self, key):
        bucket = self.hashFn(key)

        if self.table[bucket] is None:
            print("No key to be deleted.")
            return

        p = self.table[bucket]

        # If head node is the one to be deleted
        if p.data == key:
            self.table[bucket] = p.next
            print(f"Deleted [{key}] at bucket [{bucket}] (head)")
            return

        # Traverse and look for the node to delete
        while p.next is not None:
            if p.next.data == key:
                p.next = p.next.next
                print(f"Deleted [{key}] at bucket [{bucket}]")
                return
            p = p.next

        print("No key to be deleted.")


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
