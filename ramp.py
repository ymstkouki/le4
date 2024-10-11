import numpy as np

def ReLU(x):
    return np.maximum(x,0)

def fun(x):
    if x>0:
        return 1
    return 0

vfun=np.vectorize(fun)

def ReLU_bp(xlist,en_y):
    ramp_sub=vfun(xlist)
    en_x=np.multiply(en_y,ramp_sub)
    return en_x