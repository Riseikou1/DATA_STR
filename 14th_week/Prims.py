Graph = {
    'A': [('B', 29), ('F', 10)],
    'B': [('A', 29), ('C', 16), ('G', 15)],
    'C': [('B', 16), ('D', 12)],
    'D': [('C', 12), ('E', 22), ('G', 18)],
    'E': [('D', 22), ('F', 27), ('G', 25)],
    'F': [('A', 10), ('E', 27)],
    'G': [('B', 15), ('D', 18), ('E', 25)]
}

size = len(Graph)
visited = [False] * size
dist = [float('inf')] * size
parent = [-1] * size

def findMin():
    minDist = float('inf')
    minV = -1
    for v in range(size):
        if not visited[v] and dist[v] < minDist:
            minV = v
            minDist = dist[v]
    return minV

def prims(start_char):
    start = ord(start_char) - 65
    dist[start] = 0

    for _ in range(size):
        # Print current dist array
        for j in range(size):
            if dist[j] == float('inf'):
                print('  *', end=' ')
            else:
                print('%3d' % dist[j], end=' ')
        print()

        u = findMin()
        if u == -1: break  # All nodes visited or disconnected
        visited[u] = True
        u_char = chr(u + 65)
        print(f"Selected: {u_char} (dist={dist[u]})")

        for neighbor, weight in Graph[u_char]:
            v = ord(neighbor) - 65
            if not visited[v] and weight < dist[v]:
                dist[v] = weight
                parent[v] = u

    print("\nMinimum Spanning Tree:")
    total_weight = 0
    for v in range(size):
        if parent[v] != -1:
            u = parent[v]
            print(f"{chr(u+65)} - {chr(v+65)} : {dist[v]}")
            total_weight += dist[v]
    print(f"Total weight of MST: {total_weight}")

def weightSum():
    total = 0
    for v in Graph:
        for e in Graph[v]:
            total += e[1]
    return total // 2

def display():
    for v in Graph:
        for e in Graph[v]:
            if v < e[0]:
                print("[%s%s %d]" % (v, e[0], e[1]), end=" ")
        print()

if __name__ == "__main__":
    print("Original Graph Total Weight:", weightSum())
    display()
    print("\nRunning Prim's Algorithm:\n")
    prims('A')
