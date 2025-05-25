# Sample maze (6x6)
# S = Start, x = Goal, 1 = Wall, 0 = Path
# We'll replace 'S' and 'x' with '0' but start from (0,1) and aim to reach (4,4)
from stack import Arraystack

map = [
    ['1', '1', '1', '1', '1', '1'],
    ['e', '0', '0', '0', '0', '1'],
    ['1', '0', '1', '0', '1', '1'],
    ['1', '1', '1', '0', '0', 'x'],
    ['1', '1', '1', '0', '1', '1'],
    ['1', '1', '1', '1', '1', '1']
]
SIZE = 6

def isValidPos(r, c):
    if 0 <= r < SIZE and 0 <= c < SIZE:
        return map[r][c] == '0' or map[r][c] == 'x'
    return False

def DFS():
    print("DFS: ")
    stack = Arraystack(100)
    stack.push((1, 0))  # Starting position (row 1, col 0)
    map[1][0] = "."

    while not stack.isEmpty():
        here = stack.pop()  # Get the current position
        print(here, end=" -> ")
        (x, y) = here

        if map[x][y] == 'x':  # If we reach the goal
            return True
        else:
            map[x][y] = '.'  # Mark as visited
            if isValidPos(x, y - 1): stack.push((x, y - 1))  # Up
            if isValidPos(x, y + 1): stack.push((x, y + 1))  # Down
            if isValidPos(x - 1, y): stack.push((x - 1, y))  # Left
            if isValidPos(x + 1, y): stack.push((x + 1, y))  # Right

    return False

# Testing DFS
result = DFS()
if result:
    print("\n--> 미로탐색 성공")  # Maze search success
else:
    print("\n--> 미로탐색 실패")  # Maze search failure
