import mnist
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt


class Network:
    def __init__(self, class_num, ml_size):
        self.class_num = class_num
        self.ml_size = ml_size

    #mnistの画像を読み込み，0-1の範囲のベクトルに変換
    def read_mnist(self, pdir, ldir):
        X = mnist.download_and_parse_mnist_file(pdir) / 255
        self.X = X.reshape(X.shape[0], -1)
        self.Y = mnist.download_and_parse_mnist_file(ldir)
        self.pic_num = self.X.shape[0]
        self.pic_size = self.X.shape[1]

    #パラメータの初期値を与える
    #入力が変わったときのために分割(mnist -> RGB画像)
    def param_init(self, eta=0.01, batch_size=100, epoc_size=100):
        np.random.seed(0)
        self.W1 = np.random.normal(0, np.sqrt(1/self.pic_size), (self.ml_size, self.pic_size))
        self.b1 = np.random.normal(0, np.sqrt(1/self.pic_size), (self.ml_size, 1))
        self.W2 = np.random.normal(0, np.sqrt(1/self.ml_size), (self.class_num, self.ml_size))
        self.b2 = np.random.normal(0, np.sqrt(1/self.ml_size), (self.class_num, 1))

        self.eta = eta
        self.batch_size = batch_size
        self.epoc_size = epoc_size
        self.ce = np.array([])
      
    #画像を選択
    def choice(self):
        size = len(self.X)
        try:
            num = int(input("please enter an integer between 0 and " + str(size-1) + ": "))
        except ValueError:
            print("please enter an integer")
            return self.choice(self.X)
    
        if 0 <= num and num < size:
            return (num, self.X[num])
        else:
            print("please enter an appropriate number")
            return self.choice(self.X)
    

    #3層ニューラルネットワークの順伝播
    #入力層の線形和
    def fc1(self, x):
        self.fc1_out = np.dot(self.W1, x) + self.b1
    
    #中間層の活性化関数(sigmoid)
    def act1(self):
        self.act1_out = 1 / (1 + np.exp(-self.fc1_out))
    
    #中間層の線形和
    def fc2(self):
        self.fc2_out = np.dot(self.W2, self.act1_out) + self.b2
    
    #出力層の活性化関数(softmax)
    def act2(self):
        M = np.max(self.fc2_out, axis=0)
        SUM = np.sum(np.exp(self.fc2_out - M), axis=0)
        self.act2_out =  np.exp(self.fc2_out - M) / SUM
    

    #誤差逆伝播
    #act2(softmax)
    def act2_bp(self, yk):
        #ykは正解ラベルのone-hot vector
        self.en_fc2_out = np.subtract(self.act2_out, yk) / self.batch_size
    
    #fc2
    def fc2_bp(self):
        en_x = np.dot(self.W2.T, self.en_fc2_out)
        en_W2 = np.dot(self.en_fc2_out, self.act1_out.T)
        en_b2 = np.sum(self.en_fc2_out, axis=1)
        
        self.en_act1_out = en_x
        self.W2 -= self.eta * en_W2
        self.b2 -= self.eta * en_b2.reshape(en_b2.shape[0], -1)
    
    #act1(sigmoid)
    def act1_bp(self):
        tmp = np.multiply(self.act1_out, 1-self.act1_out)
        self.en_fc1_out = np.multiply(self.en_act1_out, tmp)
    
    #fc1
    def fc1_bp(self, x):
        en_W1 = np.dot(self.en_fc1_out, x.T)
        en_b1 = np.sum(self.en_fc1_out, axis=1)

        self.W1 -= self.eta * en_W1
        self.b1 -= self.eta * en_b1.reshape(en_b1.shape[0], -1)
    
    
    def cross_entropy(self, X_batch):
        e = 0
        for i in range(self.batch_size):
            t = self.Y[X_batch[i]]
            num = self.act2_out[t][i]
            e -= np.log(num) / self.batch_size
        
        self.ce = np.append(self.ce, e)
    
    #ミニバッチに対する処理
    def process_(self, X_batch, X_batch_, yk):
        self.fc1(X_batch)
        self.act1()
        self.fc2()
        self.act2()
        self.cross_entropy(X_batch_)
        self.act2_bp(yk)
        self.fc2_bp()
        self.act1_bp()
        self.fc1_bp(X_batch)
    
    #エポック数だけ繰り返し処理
    def process(self):
        for i in range(self.epoc_size):
            #エポック内で使用されていない画像番号のリスト
            self.id_list = list(range(self.X.shape[0]))
            self.ce = np.array([])

            for _ in range(self.pic_num // self.batch_size):
                X_batch_ = np.random.choice(self.id_list, self.batch_size, replace=False)
                X_batch = np.array(itemgetter(X_batch_)(self.X)).T
                #X_batch = np.array(X_batch_[self.X]).T

                #使用した画像番号を消去
                id_set = set(self.id_list) - set(X_batch_)
                self.id_list = list(id_set)

                #正解ラベルのone-hot vector表記
                yk_sub = itemgetter(X_batch_)(self.Y)
                yk = np.eye(self.class_num)[yk_sub].T

                self.process_(X_batch, X_batch_, yk)
            
            print(f'ep{i+1} : {sum(self.ce) / self.ce.shape[0]}')

    #テストデータに対する精度    
    def check_acc(self, pdir, ldir):
        X_test = mnist.download_and_parse_mnist_file(pdir) / 255
        X_test = X_test.reshape(X_test.shape[0], -1).T
        Y_test = mnist.download_and_parse_mnist_file(ldir)
        fc1_out = np.dot(self.W1, X_test) + self.b1
        act1_out = 1 / (1 + np.exp(-fc1_out))
        fc2_out = np.dot(self.W2, act1_out) + self.b2
        M = np.max(fc2_out, axis=0)
        SUM = np.sum(np.exp(fc2_out - M), axis=0)
        act2_out = np.exp(fc2_out - M) / SUM
        result = np.argmax(act2_out,axis=0)
        print(f'Accuracy : {np.count_nonzero(result == Y_test)}/{X_test.shape[1]}')
    
    #パラメータをファイルに保存
    def save_param(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    
    #パラメータをファイルから読み込み
    def load_param(self, path):
        param = np.load(path)
        self.W1 = param['W1']
        self.b1 = param['b1']
        self.W2 = param['W2']
        self.b2 = param['b2']
    
    #エントロピーの変化をプロット
    def plot_ce(self, ce_path=None):
        if ce_path != None:
            ce = np.load(ce_path)
            plt.plot(list(range(len(ce))), ce, color='r', label='cross_entropy')
        else:
            plt.plot(list(range(len(self.ce))), self.ce, color='b', label='cross_entropy')
        
        plt.legend()
        plt.show()


def main():
    #訓練用データ
    pdir_train = 'C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\train-images-idx3-ubyte.gz'
    ldir_train = 'C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\train-labels-idx1-ubyte.gz'

    #テスト用データ
    pdir_test = 'C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\t10k-images-idx3-ubyte.gz'
    ldir_test = 'C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\t10k-labels-idx1-ubyte.gz'

    n = Network(10, 50)
    n.read_mnist(pdir_train, ldir_train)
    n.param_init()
    n.process()    
    n.check_acc(pdir_test, ldir_test)


if __name__ == '__main__':
    main()
