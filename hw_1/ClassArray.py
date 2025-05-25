class Listt:

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
        

    def print(self) :
        for i in range(self.size) :
            print(f"[{i}]:{self.array[i]}")


    def reading_file(self):
        with open("/home/goku/Documents/KMU_2025_1st_Sem/datastructure/codes/hw_1/ArrayList.txt","r") as file:
            line = file.read()
            print(line)


    def changing_file(self,position, data):
        with open("/home/goku/Documents/KMU_2025_1st_Sem/datastructure/codes/hw_1/ArrayList.txt", "r") as file:
            lines = file.readlines()

        if position < len(lines):
            lines[position] = data + '\n'

            with open("/home/goku/Documents/KMU_2025_1st_Sem/datastructure/codes/hw_1/ArrayList.txt", "w") as file:
                file.writelines(lines) 

    def save(self):
        with open("list_data.txt", "w", encoding="utf-8") as file:
            for item in self.array:
                file.write(str(item)+"\n") 
        print("저장했습니다.") 