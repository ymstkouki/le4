import numpy as np

def cross_entropy(soft,Y,batch):
    e=0
    batch_size=len(batch)
    for i in range(batch_size):
        t=Y[batch[i]]
        num=soft[t][i]
        e-=np.log(num)/batch_size
    return e
