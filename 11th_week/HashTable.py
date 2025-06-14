M = 13

class HashTable:
    def __init__(self):
        self.table = [0] * M
    
    def hashFn(self,key):
        return key % M
    
    def hashFn2(self,key):
        return 11 - (key % 11)

    def insert(self,data):
        hashVal = self.hashFn(data)
        for i in range(M):
            #bucket = (hashVal+i) % M   # linera probing.
            #bucket = (hashVal+i**2) % M   # quadratic probing
            bucket = (hashVal + i * self.hashFn2(data)) % M  # Double probing
            #bucket = (hashVal)
            # umnuh linear probing ni neg dor tom cluster uusgeed baih handlagatai. meaning,future collisions more likely would happen.
            if self.table[bucket] == 0:
                self.table[bucket] = data
                break

    def search(self,data):
        hashVal = self.hashFn(data)
        for i in range(M):
            #bucket = (hasVal + i) % M    # linear probing.
            #bucket = (hashVal+i**2) % M   # quadratic probing
            bucket = (hashVal + i * self.hashFn2(data)) % M  # Double probing.
            if self.table[bucket] == 0:
                return -1
            elif self.table[bucket] == data :
                return bucket                

    def delete(self,data):
        hashVal = self.hashFn(data)
        for i in range(M):
            #bucket = (hashVal+i**2) % M   # quaratic probing
            bucket = (hashVal + i * self.hashFn2(data)) % M  # Double hashing
            if self.table[bucket] == 0:
                print("No key to delete.")
                break

            elif self.table[bucket] == data :
                self.table[bucket] = -1   # ene deer ingej hiij baigaa shaltgana ni gevel, 0 bolgochihh yum bol, search ntr deer, dund ni neg 0 shaagaad bhr,ter hurtel iteration yavad, tuuneesh tsaash baigaa sdag,olj chadku gesen ug.
                print("Deleted [%d] at bucket [%d]."%(data,bucket))
                return bucket


    def display(self):
        print("\nBucket   Data")
        print("===============")

        for i in range(M):
            print("HT[%2d] : %2d"%(i,self.table[i]))
        print()


if __name__ == "__main__":
    HT = HashTable()
    data = [45,27,88,9,71,60,46,38,24]
    for d in data :
        print("h(%2d) = %2d"%(d,HT.hashFn(d)),end=" | ")
        HT.insert(d)
        print(HT.table)

    HT.display()

    print("Search(46) ----> ",HT.search(46))

    HT.delete(9)

    HT.display()


