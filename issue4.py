import mnist
import numpy as np
from operator import itemgetter

#以下，作成したプログラムからのimport
import sigmoid as sig
import union_layer as ul
import softmax as sm
import cross_entropy as ce
import differential as dif
import batch_normalization as bn
import optimize as opt

#MNIST画像の読み込み
X=mnist.download_and_parse_mnist_file("C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\train-images-idx3-ubyte.gz")
Y=mnist.download_and_parse_mnist_file("C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\train-labels-idx1-ubyte.gz")
X=X/255

pic_num=len(X)
batch_size=100
epoc_size=100
class_num=10
ml_size=50
pic_size=28*28
entropy=[]

#学習率
eta=0.01

#"""
#重みの初期値を設定
np.random.seed(0)
W1=np.random.normal(0,np.sqrt(1/pic_size),(ml_size,pic_size))
b1=np.random.normal(0,np.sqrt(1/pic_size),(ml_size,1))
W2=np.random.normal(0,np.sqrt(1/ml_size),(class_num,ml_size))
b2=np.random.normal(0,np.sqrt(1/ml_size),(class_num,1))
"""

#重みをファイルから読み込み
W1=np.load("W1_issue3.npy")
b1=np.load("b1_issue3.npy")
W2=np.load("W2_issue3.npy")
b2=np.load("b2_issue3.npy")
"""

dW1 = np.zeros_like(W1)
s = 0
h = 1e-8
h1 = 0
h2 = 0
t1 = 0
t2 = 0
m1 = np.zeros_like(W1)
m2 = np.zeros_like(W2)
v1 = np.zeros_like(W1)
v2 = np.zeros_like(W2)

#エポック数だけ繰り返し処理
for num in range(epoc_size):
    idx_list=list(range(pic_num))

    #クロスエントロピー誤差
    cross_diff=[]

    #初期値
    gamma = 1
    beta = 0

    for i in range(pic_num//batch_size):
        X_batch=np.random.choice(idx_list,batch_size,replace=False)
        idx_set=set(idx_list)-set(X_batch)
        idx_list=list(idx_set)

        #正解ラベルのone-hot vector表記
        yk_sub=itemgetter(X_batch)(Y)
        yk=np.eye(class_num)[yk_sub].T

        Xs=np.array(itemgetter(X_batch)(X))
        X1_list=(Xs.reshape(Xs.shape[0],(Xs.shape[1]*Xs.shape[2]))).T
        Y1=ul.union(W1,X1_list,b1)
        X2_list_=sig.sigmoid(Y1)
        X2_list, x_, mu, sigma = bn.bnormalize(X2_list_, gamma, beta)
        #print(f'X2_list_:{X2_list_}, X2_list:{X2_list.shape}')
        Y2=ul.union(W2,X2_list,b2)
        results=sm.softmax(Y2)
        e=ce.cross_entropy(results,Y,X_batch)
        cross_diff.append(e)

        #誤差逆伝播
        en_soft=sm.soft_bp(results,yk,batch_size)
        #print(f'en_soft:{en_soft.shape}')
        en_x2,en_w2,en_b2=dif.diff(en_soft,W2,X2_list)
        #print(f'en_x2:{en_x2.shape}, X2_list:{X2_list.shape}')
        en_x2_, en_gamma, en_beta = bn.bnorm_bp(en_x2, X2_list, gamma, X2_list_, mu, sigma)
        en_sig=sig.sig_bp(X2_list_,en_x2_)
        _,en_w1,en_b1=dif.diff(en_sig,W1,X1_list)
        #print(f'en_w2:{sum(sum(en_w2))}')

        #重みの更新
        W1-=eta*en_w1
        #W1, dW1 = opt.MomentumSGD(W1, dW1, en_w1)
        #W1, h1 = opt.RMSProp(W1, en_w1, h1)
        b1-=eta*en_b1.reshape(ml_size,1)
        
        W2-=eta*en_w2
        #W2, h = opt.AdaGrad(W2, en_w2, h)
        #W2, h2, s = opt.AdaDelta(W2, en_w2, h2, s)
        #W2, t2, m2, v2 = opt.Adam(W2, en_w2, t2, m2, v2)
        b2-=eta*en_b2.reshape(class_num,1)
        gamma -= eta * en_gamma
        beta -= eta * en_beta
       
    print("ep" + str(num+1) + ": " + str(sum(cross_diff)/len(cross_diff)))
    entropy.append(sum(cross_diff)/len(cross_diff))

"""
#最終的な重みをファイルに保存
np.save("W1_issue3.npy", W1)
np.save("b1_issue3.npy", b1)
np.save("W2_issue3.npy", W2)
np.save("b2_issue3.npy", b2)
"""

"""
#クロスエントロピー誤差を記録
np.save("entropy_sigmoid.npy", entropy)
"""
