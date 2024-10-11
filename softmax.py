import numpy as np

def softmax(a):
    max_a=np.max(a,axis=0)
    SUM=np.sum(np.exp(a-max_a),axis=0)
    return np.exp(a-max_a)/SUM

#逆伝播
def soft_bp(results,yk,batch_size):
    en_x=np.subtract(results,yk)/batch_size
    return en_x
