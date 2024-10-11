import numpy as np

#初期値 : dW=0
def MomentumSGD(W, dW, en_W):
    eta = 0.01
    alpha = 0.9

    dW = alpha * dW + eta * en_W
    W = W + dW

    return W, dW

#初期値：h=1e-8
def AdaGrad(W, en_W, h):
    eta = 0.001

    h = h + en_W * en_W
    W = W - eta * en_W / np.sqrt(h)

    return W, h

#初期値：h=0
def RMSProp(W, en_W, h):
    eta = 0.001
    rho = 0.9
    epsilon = 1e-8

    h = rho * h + (1 - rho) * en_W * en_W
    W = W - eta + en_W / (np.sqrt(h) + epsilon)

    return W, h

#初期値：h=0, s=0
def AdaDelta(W, en_W, h, s):
    rho = 0.95
    epsilon = 1e-6

    h = rho * h + (1 - rho) * en_W * en_W
    dW = - np.sqrt(s + epsilon) / np.sqrt(h + epsilon) * en_W
    s = rho * s + (1 - rho) * dW * dW
    W = W + dW

    return W, h, s

#初期値：t=0, m=0, v=0
def Adam(W, en_W, t, m, v):
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    t = t + 1
    m = beta1 * m + (1 - beta1) * en_W
    v = beta2 * v + (1 - beta2) * en_W * en_W
    m_ = m / (1 - beta1 ** t)
    v_ = v / (1 - beta2 ** t)
    W = W - alpha * m_ / (np.sqrt(v_) + epsilon)

    return W, t, m, v
