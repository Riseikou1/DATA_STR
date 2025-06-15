Graph = {
    'A':[('B', 29), ('F', 10)],
    'B':[('A', 29),('C', 16),('G', 15)],
    'C':[('B', 16),('D', 12)],
    'D':[('C', 12),('E', 22),('G', 18)],
    'E':[('D', 22),('F', 27),('G', 25)],
    'F':[('A', 10),('E', 27)],
    'G':[('B', 15),('D', 18),('E', 25)],
}
vCnt = len(Graph)
INF = 1000
dist = [INF] * vCnt
visited = [False] * vCnt

def findMin():
    minDist = INF
    minV = 0

    for v in range(vCnt):
        if visited[v] == False and dist[v] < minDist:
            minDist = dist[v]
            minV = v
    return minV     #6

def prim(start):
    dist[ord(start)-65] = 0

    for _ in range(vCnt):
        for j in range(vCnt):
            print(" * " if dist[j] == INF else "%3d " % dist[j], end="")
        print()

        u = findMin()
        visited[u] = True
        uName = chr(u + 65)

        for neighbor, weight in Graph[uName]:
            v = ord(neighbor) - 65
            if not visited[v] and weight < dist[v]:
                dist[v] = weight


if __name__ == '__main__':
    prim('G')