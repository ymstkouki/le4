def choice(X):
    size=len(X)
    try:
        num=int(input("please enter an integer between 0 and " + str(size-1) + ": "))
    except ValueError:
        print("please enter an integer")
        return choice(X)
    
    if 0<=num and num<size:
        return (num, X[num])
    else:
        print("please enter an appropriate number")
        return choice(X)
