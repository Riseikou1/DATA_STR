INF = float("inf")

vName = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
Graph = [
    [ 0, 7, INF, INF, 3, 10, INF],
    [ 7, 0, 4, 10, 2, 6, INF],
    [INF, 4, 0, 2, INF, INF, INF],
    [INF, 10, 2, 0, 11, 9, 4],
    [ 3, 2, INF, 11, 0, 13, 5],
    [ 10, 6, INF, 9, 13, 0, INF],
    [INF, INF, INF, 4, 5, INF, 0]
]

vCnt = len(vName)

dist = [INF] * vCnt   # 7 
visited = [False] * vCnt  # 7

def findMin():
    minDist = INF
    minV = 0

    for v in range(vCnt):
        if visited[v] == False and dist[v] < minDist:
            minDist = dist[v]
            minV = v
    return minV    

def display():

    for i in range(vCnt):
        if dist[i] == INF:
            print(' ∞ ', end='')
        else:
            print('%2d ' % dist[i], end='')
    print()

def dijkstra(s):
    dist[ord(s)-ord('A')] = 0

    for _ in range(vCnt):
        s = findMin()
        if s == -1 : break
        visited[s] = True

        for t in range(vCnt):
            if dist[t] > dist[s] + Graph[s][t]:
                dist[t] = dist[s] + Graph[s][t]

        print('[%c] : ' % vName[s], end='')
        display()

if __name__ == '__main__':
    dijkstra('A')