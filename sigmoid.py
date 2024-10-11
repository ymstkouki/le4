import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

#逆伝播
def sig_bp(ylist,en_y):
    y_list=np.array(ylist)
    sig_sub=np.multiply(y_list,1-y_list)
    en_x=np.multiply(en_y,sig_sub)
    return en_x
