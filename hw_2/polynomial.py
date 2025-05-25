class Poly:
    def __init__(self, capacity=10):
        self.degree = 0
        self.coef = [0] * capacity

    def readPoly(self):
        self.degree = int(input("다항식의 차수를 입력하세요: "))
        if len(self.coef) <= self.degree:
            self.coef += [0] * (self.degree - len(self.coef) + 1)

        for i in range(self.degree, 0, -1):
            c = int(input(f"x^{i}의 계수를 입력하세요: "))
            self.coef[i] = c
        self.coef[0] = int(input("상수항을 입력하세요: "))

    def printPoly(self):
        strr = ""
        for i in range(self.degree, 0, -1):
            coef = self.coef[i]
            if coef != 0:
                sign = " + " if coef > 0 and strr else (" - " if coef < 0 and strr else ("-" if coef < 0 else ""))
                abs_coef = abs(coef)
                if i == 1:
                    strr += f"{sign}{abs_coef}x"
                else:
                    strr += f"{sign}{abs_coef}x^{i}"
        if self.coef[0] != 0:
            sign = " + " if self.coef[0] > 0 and strr else (" - " if self.coef[0] < 0 and strr else ("-" if self.coef[0] < 0 else ""))
            strr += f"{sign}{abs(self.coef[0])}"

        print(f"다항식: {strr}\n")

    def eval(self):
        result = 0
        value = int(input("x의 값을 입력하세요: "))
        for i in range(self.degree, 0, -1):
            result += (value ** i) * self.coef[i]

        result += self.coef[0]
        print(f"다항식 계산 결과: {result}\n")


    def add(self):
        print("두 개의 다항식을 더합니다.\n")
        second = Poly()
        second.readPoly()

        max_deg = self.degree if self.degree > second.degree else second.degree

        a_coef = self.coef + [0] * (max_deg - self.degree)
        b_coef = second.coef + [0] * (max_deg - second.degree)

        result_coef = [a_coef[i] + b_coef[i] for i in range(max_deg + 1)]

        strr = ""
        for i in range(max_deg, 0, -1):
            coef = result_coef[i]
            if coef != 0:
                sign = " + " if coef > 0 and strr else (" - " if coef < 0 and strr else ("-" if coef < 0 else ""))
                abs_coef = abs(coef)
                if i == 1:
                    strr += f"{sign}{abs_coef}x"
                else:
                    strr += f"{sign}{abs_coef}x^{i}"

        if result_coef[0] != 0:
            sign = " + " if result_coef[0] > 0 and strr else (" - " if result_coef[0] < 0 and strr else ("-" if result_coef[0] < 0 else ""))
            strr += f"{sign}{abs(result_coef[0])}"

        print(f"두 다항식의 합: {strr}\n")


if __name__ == "__main__":
    a = Poly()
    a.readPoly()
    a.printPoly()
    a.eval()
    a.add()

