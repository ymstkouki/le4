import numpy as np

def diff(en_y, w, x):
    en_x=np.dot(w.T, en_y)
    en_w=np.dot(en_y, x.T)
    en_b=np.sum(en_y, axis=1)
    return (en_x, en_w, en_b)