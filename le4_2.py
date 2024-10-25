import numpy as np
from le4_1 import Network

#活性化関数
#sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ReLU
def ReLU(x):
    return np.maximum(x, 0)

#dropout
def dropout(x, rho, istest=False):
    if istest:
        return (1 - rho) * x
    else:
        ignores = []
        k = int(np.floor(len(x) * rho))
        ids = list(range(len(x)))
        for j in range(x.shape[1]):
            ignore = np.random.choice(ids, k, replace=False)
            for i in ignore:
                x[i][j] = 0
            ignores.append(ignore)
        return x, ignores


#逆伝播
#sigmoid
def sigmoid_bp(y, en_y):
    y = np.array(y)
    sig_sub = np.multiply(y, 1-y)
    en_x = np.multiply(en_y, sig_sub)
    return en_x

#ReLU
def ReLU_bp_(x):
    en_x_ = 0
    if x > 0:
        en_x_ = 1
    return en_x_

vfun = np.vectorize(ReLU_bp_)

def ReLU_bp(x, en_y):
    ReLU_sub = vfun(x)
    en_x = np.multiply(en_y, ReLU_sub)
    return en_x

#dropout
def dropout_bp(x, en_y, ignores):
    drop_sub = np.ones(x.shape)
    for i in range(len(ignores)):
        for j in ignores[i]:
            drop_sub[j][i] = 0
    en_x = np.multiply(drop_sub, en_y)
    return en_x


class CustomizedNetwork(Network):
    def __init__(self, class_num, ml_size, opt1):
        super().__init__(class_num, ml_size)
        self.opt1 = opt1
    
    def act1(self):
        if 's' in self.opt1:
            self.act1_out = sigmoid(self.fc1_out)
        elif 'r' in self.opt1:
            self.act1_out = ReLU(self.fc1_out)
    
    def act1_bp(self):
        if 's' in self.opt1:
            self.en_fc1_out = sigmoid_bp(self.act1_out, self.en_act1_out)
        elif 'r' in self.opt1:
            self.en_fc1_out = ReLU_bp(self.fc1_out, self.en_act1_out)

def main():
    #訓練用データ
    pdir_train = 'C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\train-images-idx3-ubyte.gz'
    ldir_train = 'C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\train-labels-idx1-ubyte.gz'

    #テスト用データ
    pdir_test = 'C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\t10k-images-idx3-ubyte.gz'
    ldir_test = 'C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\t10k-labels-idx1-ubyte.gz'

    cn = CustomizedNetwork(10, 50, 's')
    cn.read_mnist(pdir_train, ldir_train)
    cn.param_init(epoc_size=100)
    cn.process()    
    cn.check_acc(pdir_test, ldir_test)

if __name__ == '__main__':
    main()
