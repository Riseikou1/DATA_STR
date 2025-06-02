vName = ['A','B','C','D','E','F','G','H']
visited = [False]*len(vName)

Graph = [[1,2],[0,3],[0,3,4],[1,2,5],[2,6,7],[3],[4,7],[4,6]]

from queue import LifoQueue

class Stack(LifoQueue):   # doing this cuz there's no peek method in lifoSueue.
    def peek(self):
        if not self.empty():
            return self.queue[-1]
        raise Exception("Empty")
    

def iDfs(s : int) :
    S = Stack()
    S.put(s)
    visited[s] = True
    print("[%c] " % vName[s],end="")

    while not S.empty():
        s = S.peek()
        flag = False
        for t in Graph[s]:
            if not visited[t] :
                S.put(t)
                visited[t] = True
                print("[%c] " % vName[t],end="")
                flag = True
                break
                
        if not flag : S.get()
        
if __name__ == "__main__":
    print("iDFS : ", end="")
    iDfs(1) ; print()

