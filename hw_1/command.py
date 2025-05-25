from ClassArray import Listt

list = Listt()


while True:

    command = input("[메뉴선택]: i-입력, d-삭제, r-변경, p-출력, l-파일읽기, s-저장, q-종료 ==> ")

    match command :
        case 'i' :
            pos = int(input("입력행 번호: "))
            str = input("입력행 내용: ")
            list.insert(pos,str)

        case 'd' :
            pos = int(input("삭제행 번호: "))
            list.delete(pos)

        case 'r' :
            position = int(input("변경행 번호: "))
            data = input("변경행 내용: ")
            list.changing_file(position,data)

        case 'p' :
            list.print()

        case 'l' :
            list.reading_file()
        
        case 's':
            list.save()
        
        case 'q':
            print("시스템 종료합니다.")
            break



