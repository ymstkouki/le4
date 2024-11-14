import numpy as np
import pickle
import random
from operator import itemgetter
import matplotlib.pyplot as plt
from tqdm import tqdm

from le4_1 import Network

#入力(畳み込み→プーリング)→中間(全結合)→出力
class ColorImageNetwork(Network):
    def __init__(self, class_num, filter_size, filter_ch):
        self.class_num = class_num
        self.filter_size = filter_size
        self.filter_ch = filter_ch

    def param_init(self, eta=0.01, batch_size=100, epoc_size=100, pool=0):
        np.random.seed(1)

        #畳み込み
        self.W1 = np.random.normal(0, np.sqrt(1/3*self.filter_size**2), (self.filter_ch, 3*self.filter_size**2))
        self.b1 = np.random.normal(0, np.sqrt(1/self.pic_sizex*self.pic_sizey), (self.filter_ch, 1))
        
        #全結合
        if pool == 0:
            self.W2 = np.random.normal(0, np.sqrt(1/self.filter_ch*self.pic_sizex*self.pic_sizey), (self.class_num, self.filter_ch*self.pic_sizex*self.pic_sizey))
        else:
            self.W2 = np.random.normal(0, np.sqrt(1/self.filter_ch*self.pic_sizex*self.pic_sizey), (self.class_num, self.filter_ch*(self.pic_sizex//pool)*(self.pic_sizey//pool)))
        self.b2 = np.random.normal(0, np.sqrt(1/self.filter_ch*self.pic_sizex*self.pic_sizey), (self.class_num, 1))
        
        self.eta = eta
        self.batch_size = batch_size
        self.epoc_size = epoc_size
        self.pool = pool
        self.ce = np.array([])

    #カラー画像の読み込み
    #dirはリスト
    def read_cifar(self, dir):
        num = len(dir)

        #1つのファイルに10000枚の画像データが格納されていることを想定
        pics = 10000
        self.pic = np.empty((pics*num, 3, 32, 32))
        self.label = np.empty(pics*num, dtype=np.int8)

        for i in range(num):
            with open(dir[i], 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                pic = np.array(dict[b'data'])

            #枚数，色数，縦ピクセル数，横ピクセル数
            self.pic[pics*i:pics*(i+1), :, :, :] = pic.reshape(pics, 3, 32, 32) / 255
            self.label[pics*i:pics*(i+1)] = np.array(dict[b'labels'])

        self.pic_num = self.pic.shape[0]
        self.pic_sizex = self.pic.shape[3]
        self.pic_sizey = self.pic.shape[2]
    
    #フィルタサイズに応じたパディング
    #ミニバッチごとに実行
    def padding(self):
        pad_size = self.filter_size // 2
        self.pad_size = pad_size
        padded_pic = np.zeros((self.batch_size, self.pic.shape[1], self.pic.shape[2]+2*pad_size, self.pic.shape[3]+2*pad_size))
        padded_pic[:, :, pad_size:-pad_size, pad_size:-pad_size] = self.batch_pic
        self.padded_pic = padded_pic
    
    #パディング済みの画像から行列Xに変換
    #画像数はそのまま
    #ここが見直し必要かも
    def padded2matrix(self):
        X = np.empty((self.padded_pic.shape[0], 3, self.filter_size, self.filter_size, self.pic_sizex, self.pic_sizey))
        for i in (range(self.filter_size)):
            for j in range(self.filter_size):
                X[:, :, i, j, :, :] = self.padded_pic[:, :, i:i+self.pic_sizey, j:j+self.pic_sizex]
                # X[:, i*self.filter_size+j, :] = self.padded_pic[:, 0, i:i+self.pic_sizey, j:j+self.pic_sizex].reshape(-1, self.pic_sizex*self.pic_sizey)
                # X[:, i*self.filter_size+j+self.filter_size**2, :] = self.padded_pic[:, 1, i:i+self.pic_sizey, j:j+self.pic_sizex].reshape(-1, self.pic_sizex*self.pic_sizey)
                # X[:, i*self.filter_size+j+2*self.filter_size**2, :] = self.padded_pic[:, 2, i:i+self.pic_sizey, j:j+self.pic_sizex].reshape(-1, self.pic_sizex*self.pic_sizey)
        # batch_X = X.transpose(1, 0, 2)
        # print(27*100*1024 - np.count_nonzero(batch_X))
        # self.batch_X = batch_X.reshape(batch_X.shape[0], batch_X.shape[1]*batch_X.shape[2])
        self.batch_X = X.transpose(0, 4, 5, 1, 2, 3).reshape(X.shape[0]*self.pic_sizex*self.pic_sizey, -1).T

    #処理済みのW，X，Bを使って畳み込みの計算
    #最終的には(-1, batch_size)の形で与える
    def convolve(self):
        conv_out = np.dot(self.W1, self.batch_X) + self.b1
        conv_out = conv_out.T
        
        #プーリング層を使う場合
        if self.pool != 0:
            conv_out = conv_out.reshape(self.batch_size, self.pic_sizex, self.pic_sizey, self.filter_ch)
            self.conv_out = conv_out.transpose(0, 3, 1, 2)
        #プーリング層を使わない場合
        else:
            conv_out = conv_out.reshape(self.batch_size, -1, self.filter_ch)
            conv_out = conv_out.transpose(0, 2, 1)
            conv_out = conv_out.reshape(self.batch_size, -1)
            self.conv_out = conv_out.T
    
    #逆伝播
    def convolve_bp(self):
        en_conv_out = self.en_conv_out.T
        en_conv_out = en_conv_out.reshape(self.batch_size, self.filter_ch, -1)
        en_conv_out = en_conv_out.transpose(0, 2, 1)
        en_conv_out = en_conv_out.reshape(-1, self.filter_ch)
        self.en_conv_out = en_conv_out.T

        en_W1 = np.dot(self.en_conv_out, self.batch_X.T)
        en_b1 = np.sum(self.en_conv_out, axis=1).reshape(-1, 1)

        self.W1 -= self.eta * en_W1
        self.b1 -= self.eta * en_b1

    #畳み込み層の活性化関数(sigmoid)
    def act1(self):
        self.act1_out = 1 / (1 + np.exp(-self.conv_out))
    
    #逆伝播
    def act1_bp(self):
        tmp = np.multiply(self.act1_out, 1-self.act1_out)
        self.en_conv_out = np.multiply(self.en_act1_out, tmp)
    
    #プーリング層
    def pooling(self):
        tmp = self.act1_out
        d = self.pool
        pooling = np.empty((tmp.shape[0], tmp.shape[1], tmp.shape[2]//d, tmp.shape[3]//d))
        ind = np.empty((tmp.shape[0], tmp.shape[1], tmp.shape[2]//d, tmp.shape[3]//d, 2), dtype=np.int8)
        for i in range(pooling.shape[0]):
            for j in range(pooling.shape[1]):
                for k in range(pooling.shape[2]):
                    for l in range(pooling.shape[3]):
                        window = tmp[i, j, d*k:d*(k+1), d*l:d*(l+1)]
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        pooling[i, j, k, l] = window[max_pos]
                        ind[i, j, k, l] = (k+max_pos[0], l+max_pos[1])
        self.pooling_out = pooling.reshape(pooling.shape[0], -1).T
        self.pooling_ind = ind
    
    #逆伝播
    def pooling_bp(self):
        d = self.pool
        en_act1_out = np.zeros_like(self.act1_out)
        for i in range(en_act1_out.shape[0]):
            for j in range(en_act1_out.shape[1]):
                for k in range(en_act1_out.shape[2]//d):
                    for l in range(en_act1_out.shape[3]//d):
                        max_pos = self.pooling_ind[i, j, k, l]
                        en_act1_out[i, j, max_pos[0], max_pos[1]] = self.en_pooling_out[j*(en_act1_out.shape[2]//d)*(en_act1_out.shape[3]//d)+k*(en_act1_out.shape[3]//d)+l, i]
        self.en_act1_out = en_act1_out
    
    #プーリング層からの出力を全結合
    def fc2_pool(self):
        self.fc2_out = np.dot(self.W2, self.pooling_out) + self.b2

    #逆伝播
    def fc2_pool_bp(self):
        en_x = np.dot(self.W2.T, self.en_fc2_out)
        en_W2 = np.dot(self.en_fc2_out, self.pooling_out.T)
        en_b2 = np.sum(self.en_fc2_out, axis=1)
        
        self.en_pooling_out = en_x
        self.W2 -= self.eta * en_W2
        self.b2 -= self.eta * en_b2.reshape(en_b2.shape[0], -1)
    
    def act2_bp(self):
        #ykは正解ラベルのone-hot vector
        self.en_fc2_out = np.subtract(self.act2_out, self.yk) / self.batch_size
    
    def cross_entropy(self):
        #devide by zero in logを回避
        epsilon = 1e-15
        e = 0

        for i in range(self.batch_size):
            t = self.label[self.batch_ind[i]]
            num = self.act2_out[t, i] + epsilon
            e -= np.log(num) / self.batch_size
        self.ce = np.append(self.ce, e)

    def check_acc(self):
        result = np.argmax(self.act2_out, axis=0)
        return np.count_nonzero(result == self.label[self.batch_ind])

    #ミニバッチに対する処理
    def process_(self):
        self.convolve()
        self.act1()
        if self.pool != 0:
            self.pooling()
            self.fc2_pool()
        else:
            self.fc2()
        self.act2()
        self.cross_entropy()
        self.act2_bp()
        if self.pool != 0:
            self.fc2_pool_bp()
            self.pooling_bp()
        else:
            self.fc2_bp()
        self.act1_bp()
        self.convolve_bp()

    def process(self):
        for i in (range(self.epoc_size)):
            #エポック内で使用されていない画像番号のリスト
            self.id_list = list(range(self.pic.shape[0]))
            self.ce = np.array([])

            count = 0
            for _ in tqdm(range(self.pic_num // self.batch_size)):
                batch_ind = np.random.choice(self.id_list, self.batch_size, replace=False)
                self.batch_ind = batch_ind
                batch_pic = np.array(itemgetter(batch_ind)(self.pic))
                self.batch_pic = batch_pic
                self.padding()
                self.padded2matrix()

                #使用した画像番号を消去
                id_set = set(self.id_list) - set(batch_ind)
                self.id_list = list(id_set)

                #正解ラベルのone-hot vector表記
                yk_sub = itemgetter(batch_ind)(self.label)
                self.yk_sub = yk_sub
                yk = np.eye(self.class_num)[yk_sub].T
                self.yk = yk

                self.process_()
                count += self.check_acc()
            
            print(f'ep{i+1} : {sum(self.ce) / self.ce.shape[0]}')
            print(count/50000)
            
    
def main():
    cin = ColorImageNetwork(10, 3, 5)
    cin.read_cifar(["C:\\Users\\yamashita kouki\\Downloads\\cifar-10-python\\cifar-10-batches-py\\data_batch_1", "C:\\Users\\yamashita kouki\\Downloads\\cifar-10-python\\cifar-10-batches-py\\data_batch_2", "C:\\Users\\yamashita kouki\\Downloads\\cifar-10-python\\cifar-10-batches-py\\data_batch_3", "C:\\Users\\yamashita kouki\\Downloads\\cifar-10-python\\cifar-10-batches-py\\data_batch_4", "C:\\Users\\yamashita kouki\\Downloads\\cifar-10-python\\cifar-10-batches-py\\data_batch_5"])
    cin.param_init()
    cin.process()
    cin.plot_ce()


if __name__ == '__main__':
    main()
