Graph = {
    'A': [('B', 29), ('F', 10)],
    'B': [('A', 29), ('C', 16), ('G', 15)],
    'C': [('B', 16), ('D', 12)],
    'D': [('C', 12), ('E', 22), ('G', 18)],
    'E': [('D', 22), ('F', 27), ('G', 25)],
    'F': [('A', 10), ('E', 27)],
    'G': [('B', 15), ('D', 18), ('E', 25)]
}

vName = sorted(Graph.keys())  # ['A', 'B', 'C', 'D', 'E', 'F', 'G']
vCnt = len(vName)
index = {v: i for i, v in enumerate(vName)}  # map node name to index
rev_index = {i: v for v, i in index.items()}  # reverse lookup

visited = [False] * vCnt
dist = [float('inf')] * vCnt

def findMin():
    minDist = float('inf')
    minV = -1
    for v in range(vCnt):
        if not visited[v] and dist[v] < minDist:
            minV = v
            minDist = dist[v]
    return minV

def display():
    for d in dist:
        if d == float('inf'):
            print(' ∞ ', end='')
        else:
            print('%2d' % d, end=' ')
    print()

def dijkstra(start):
    dist[index[start]] = 0
    for _ in range(vCnt):
        u = findMin()
        if u == -1:
            break
        visited[u] = True
        u_name = rev_index[u]

        for v_name, weight in Graph[u_name]:
            v = index[v_name]
            if not visited[v] and dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight

        print('[%c] : ' % u_name, end='')
        display()

if __name__ == "__main__":
    dijkstra('A')
