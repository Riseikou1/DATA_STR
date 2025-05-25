class Car :
    def __init__(self,color,speed=0):
        self.color = color
        self.speed = speed

    def speedUp(self):
        self.speed += 10

    def speedDown(self):
        self.speed -= 10

    def __eq__(self,carB):
        return self.color == carB.color

    def __str__(self):
        return "color = %s, speed=%d"%(self.color,self.speed)

car1 = Car("black",100)
car2 = Car("black",50)

car1.speedUp()
print(car1==car2)

print("[car1]=> ",car1)



class SuperCar(Car):
    def __init__(self,color,speed=0,bTurbo=True):
        super().__init__(color,speed)
        self.bTurbo = bTurbo

    def speedUp(self):
        if self.bTurbo :
            self.speed += 50
        else :
            super().speedUp()

    def __str__(self):
        if self.bTurbo :
            return "color = %s, speed=%d , Turbo Mode is ON!!!"%(self.color,self.speed)
        else :
            return super().__str__()

s1 = SuperCar("Gold",0,True)
s2 = SuperCar("White",0,False)
s1.speedUp()
s2.speedUp()
print("superCar1: ",s1)
print("superCar2: ",s2)

