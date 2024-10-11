import numpy as np

def drop(x,rho,istest=False):
    if istest:
        return (1-rho)*x
    
    ignores=[]
    k=int(np.floor(len(x)*rho))
    ids=list(range(len(x)))
    for j in range(x.shape[1]):
        ignore=np.random.choice(ids,k,replace=False)
        for i in ignore:
            x[i][j]=0
        ignores.append(ignore)
    return x,ignores

def drop_bp(xlist,en_y,ignores):
    drop_sub=np.ones(xlist.shape)
    for i in range(len(ignores)):
        for j in ignores[i]:
            drop_sub[j][i]=0
    en_x=np.multiply(drop_sub,en_y)
    return en_x