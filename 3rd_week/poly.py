class Poly :

    def __init__(self,capacity = 10):
        self.capacity = capacity
        self.degree = 0
        self.coef = [None]*capacity

    def readPoly(self):
        self.degree = int(input("input polynomial shits : "))
        for i in range(self.degree,-1,-1) :
            c = int(input("%d차 항의 계수: " %i))
            self.coef[i] = c
        
    def printPoly(self):
        for i in range(self.degree,0,-1) :
            if(i!=1):   # x^1 power gej print shaaval teneg bolohoor.
                print("%dx^%d + " % (self.coef[i],i),end="")
            else :
                print("%dx + " % (self.coef[i]),end="")
        print(self.coef[0])

    
if __name__ == "__main__":
    a = Poly()
    a.readPoly()
    a.printPoly()