def hanoi_tower(n,fr,tmp,to):
    if(n==1):
        print("Circle 1: %s --> %s" %(fr,to))
    else:
        hanoi_tower(n-1,fr,to,tmp)
        print("Circle %d: %s --> %s" %(n,fr,to))
        hanoi_tower(n-1,tmp,fr,to)

hanoi_tower(4,'A','B','C')
