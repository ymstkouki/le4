import mnist
import numpy as np
from operator import itemgetter

#以下，作成したプログラムからのimport
import input_layer as il
import sigmoid as sig
import union_layer as ul
import softmax as sm
import cross_entropy as ce

pic_size=28*28
batch_size=100
class_num=10
ml_size=100

#MNIST画像の読み込み
X=mnist.download_and_parse_mnist_file("C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\train-images-idx3-ubyte.gz")
Y=mnist.download_and_parse_mnist_file("C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\train-labels-idx1-ubyte.gz")
X=X/255

#ミニバッチの取り出し
np.random.seed(0)
X_batch=np.random.choice(len(X),batch_size,replace=False)

#重みの初期値を設定
W1=np.random.normal(0,np.sqrt(1/pic_size),(ml_size,pic_size))
b1=np.random.normal(0,np.sqrt(1/pic_size),(ml_size,batch_size))
W2=np.random.normal(0,np.sqrt(1/ml_size),(class_num,ml_size))
b2=np.random.normal(0,np.sqrt(1/ml_size),(class_num,batch_size))

results=[]

Xs=np.array(itemgetter(X_batch)(X))
Xs=Xs.reshape(Xs.shape[0],(Xs.shape[1]*Xs.shape[2]))
union_X0=ul.union(W1,Xs.T,b1)
y1=sig.sigmoid(union_X0)
a=ul.union(W2,y1,b2)
soft_a=sm.softmax(a)
e=ce.cross_entropy(soft_a,Y,X_batch)
print(e)

"""
#ミニバッチに対するニューラルネットワークの計算
for j in range(batch_size):
    X0=X[X_batch[j]]
    flat_X0=il.flat(X0)
    union_X0=ul.union(W1,flat_X0,b1)
    y1=sig.sigmoid(union_X0)
    a=ul.union(W2,y1,b2)
    soft_a=sm.softmax(a)
    results.append(soft_a)

#クロスエントロピー誤差の計算
e=ce.cross_entropy(results,Y,X_batch)
print(e)
"""