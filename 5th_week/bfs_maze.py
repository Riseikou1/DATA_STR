from circular_queue import CircularQueue

map = [
#0    #1   #2   #3   #4   #5  
['1', '1', '1', '1', '1', '1'], # 0
['e', '0', '1', '0', '0', '1'], # 1
['1', '0', '0', '0', '1', '1'], # 2
['1', '0', '1', '0', '1', '1'], # 3
['1', '0', '1', '0', '0', 'x'], # 4
['1', '1', '1', '1', '1', '1']  # 5
]

SIZE = 6


def isValidPos(r, c):
    if 0 <= r < SIZE and 0 <= c < SIZE:
        return map[r][c] == '0' or map[r][c] == 'x'
    return False

def BFS():
    print("BFS: ")
    Q = CircularQueue()
    Q.enqueue((1,0))
    map[1][0] = "."

    while not Q.isEmpty():
        pos = Q.dequeue()
        print(pos, end=" -> ")
        (r, c) = pos

        if map[r][c] == 'x':
            return True
        else:
            map[r][c] = '.'  
            if isValidPos(r, c-1): Q.enqueue((r, c-1))  
            if isValidPos(r, c+1): Q.enqueue((r, c+1))  
            if isValidPos(r-1, c): Q.enqueue((r-1, c))
            if isValidPos(r+1, c): Q.enqueue((r+1, c))

    return False

result = BFS()
if result:
    print("\n--> 미로탐색 성공")
else:
    print("\n--> 미로탐색 실패")
