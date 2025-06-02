vName = ['A','B','C','D','E','F','G','H']
visited = [False]*len(vName)

Graph = [[0,1,1,0,0,0,0,0],
         [1,0,0,1,0,0,0,0],
         [1,0,0,1,1,0,0,0],
         [0,1,1,0,0,1,0,0],
         [0,0,1,0,0,0,1,1],
         [0,0,0,1,0,0,0,0],
         [0,0,0,0,1,0,0,1],
         [0,0,0,0,1,0,1,0]]


def rDFS(s):
    visited[s] = True
    print('[%c] '% vName[s], end='')

    for t in range(len(vName)):
        if not visited[t] and Graph[s][t] == 1:
            rDFS(t)


if __name__ == "__main__":
    print('rDFS : ',end="")
    rDFS(1) ; print()



