import numpy as np
import pickle
import random
from operator import itemgetter
import matplotlib.pyplot as plt
from tqdm import tqdm

from le4_1 import Network

"""
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
        X = np.array(dict[b'data'])
        X = X.reshape((X.shape[0],3,32,32))
        Y = np.array(dict[b'labels'])
    return X,Y

X,Y = unpickle("C:\\Users\\yamashita kouki\\Downloads\\cifar-10-python\\cifar-10-batches-py\\data_batch_1")
idx = 1000
plt.imshow(X[idx].transpose((1,2,0))) # X[idx] が (3*32*32) になっているのを (32*32*3) に変更する．
plt.show() # トラックの画像が表示されるはず
print(f'{Y[idx]=}') # 9 番（truck）が表示されるはず
"""

class ColorImageNetwork(Network):
    def __init__(self, class_num, filter_size, filter_ch):
        self.class_num = class_num
        self.filter_size = filter_size
        self.filter_ch = filter_ch

    def param_init(self, eta=0.01, batch_size=100, epoc_size=100):
        np.random.seed(0)
        self.W1 = np.random.normal(0, np.sqrt(1/self.pic_sizex*self.pic_sizey), (self.filter_ch, 3*self.filter_size**2))
        self.b1 = np.random.normal(0, np.sqrt(1/self.pic_sizex*self.pic_sizey), (1, batch_size*self.pic_sizex*self.pic_sizey))
        self.W2 = np.random.normal(0, np.sqrt(1/self.filter_ch), (self.class_num, self.filter_ch*self.pic_sizex*self.pic_sizey))
        self.b2 = np.random.normal(0, np.sqrt(1/self.filter_ch), (self.class_num, 1))

        self.eta = eta
        self.batch_size = batch_size
        self.epoc_size = epoc_size
        self.ce = np.array([])

    #カラー画像の読み込み
    def read_cifar(self, dir):
        with open(dir, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            pic = np.array(dict[b'data'])
            self.pic = pic.reshape(pic.shape[0], 3, 32, 32) / 255
            self.label = np.array(dict[b'labels'])
        self.pic_num = self.pic.shape[0]
        self.pic_sizex = self.pic.shape[2]
        self.pic_sizey = self.pic.shape[3]
    
    #フィルタサイズに応じたパディング
    def padding(self):
        pad_size = self.filter_size // 2
        self.pad_size = pad_size
        padded_pic = np.zeros((self.pic.shape[0], self.pic.shape[1], self.pic.shape[2]+2*pad_size, self.pic.shape[3]+2*pad_size))
        padded_pic[:, :, pad_size:-pad_size, pad_size:-pad_size] = self.pic
        self.padded_pic = padded_pic
    
    #パディング済みの画像から行列Xに変換
    def padded2matrix(self):
        self.X = np.empty((self.padded_pic.shape[0], self.pic_sizex*self.pic_sizey, 3*self.filter_size*self.filter_size))
        for i in tqdm(range(self.padded_pic.shape[2]-2*self.pad_size)):
            for j in range(self.padded_pic.shape[3]-2*self.pad_size):
                self.X[:, i*32+j, :] = self.padded_pic[:, :, i:i+self.filter_size, j:j+self.filter_size].reshape(-1, self.padded_pic.shape[1]*self.filter_size*self.filter_size)
    
    #処理済みのW，X，Bを使って畳み込みの計算
    def convolve(self, X):
        self.conv_out = np.dot(self.W1, X) + self.b1
        self.conv_out = self.conv_out.reshape(self.filter_ch, -1, self.batch_size)
        self.conv_out = self.conv_out.reshape(-1, self.batch_size)
    
    #逆伝播，en_conv_outが必要
    def convolve_bp(self, X):
        self.en_conv_out = self.en_conv_out.reshape(self.filter_ch, -1)
        en_W1 = np.dot(self.en_conv_out, X.T)
        en_b1 = np.sum(self.en_conv_out)

        self.W1 -= self.eta * en_W1
        self.b1 -= self.eta * en_b1

    #中間層の活性化関数(sigmoid)
    def act1(self):
        self.act1_out = 1 / (1 + np.exp(-self.conv_out))
    
    #act1(sigmoid)
    def act1_bp(self):
        tmp = np.multiply(self.act1_out, 1-self.act1_out)
        self.en_conv_out = np.multiply(self.en_act1_out, tmp)
    
    def cross_entropy(self, X_batch):
        e = 0
        for i in range(self.batch_size):
            t = self.label[X_batch[i]]
            num = self.act2_out[t, i]
            e -= np.log(num) / self.batch_size
        
        self.ce = np.append(self.ce, e)

    #ミニバッチに対する処理
    def process_(self, X_batch, X_batch_, yk):
        self.convolve(X_batch)
        self.act1()
        self.fc2()
        self.act2()
        self.cross_entropy(X_batch_)
        self.act2_bp(yk)
        self.fc2_bp()
        self.act1_bp()
        self.convolve_bp(X_batch)

    def process(self):
        for i in tqdm(range(self.epoc_size)):
            #エポック内で使用されていない画像番号のリスト
            self.id_list = list(range(self.pic.shape[0]))
            self.ce = np.array([])

            for _ in range(self.pic_num // self.batch_size):
                X_batch_ = np.random.choice(self.id_list, self.batch_size, replace=False)
                X_batch = np.array(itemgetter(X_batch_)(self.X)).T
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1]*X_batch.shape[2])

                #使用した画像番号を消去
                id_set = set(self.id_list) - set(X_batch_)
                self.id_list = list(id_set)

                #正解ラベルのone-hot vector表記
                yk_sub = itemgetter(X_batch_)(self.label)
                yk = np.eye(self.class_num)[yk_sub].T

                self.process_(X_batch, X_batch_, yk)
            
            print(f'ep{i+1} : {sum(self.ce) / self.ce.shape[0]}')
    
    def save_X(self, path):
        np.savez(path, X=self.X)
    
    def load_X(self, path):
        data = np.load(path)
        self.X = data['X']
    
def main():
    cin = ColorImageNetwork(10, 3, 5)
    cin.read_cifar("C:\\Users\\yamashita kouki\\Downloads\\cifar-10-python\\cifar-10-batches-py\\data_batch_1")
    cin.param_init()
    cin.padding()
    cin.padded2matrix()
    cin.process()
    cin.plot_ce()


if __name__ == '__main__':
    main()
