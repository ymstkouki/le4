import mnist
import numpy as np

#以下，作成したプログラムからのimport
import pic_choice as pc
import input_layer as il
import sigmoid as sig
import union_layer as ul
import softmax as sm
import decision as dec


pic_size=28*28
class_num=10
ml_size=100

#MNIST画像の読み込み
X=mnist.download_and_parse_mnist_file("C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\t10k-images-idx3-ubyte.gz")
X=X/255

#重みの初期値を設定
np.random.seed(0)
W1=np.random.normal(0,np.sqrt(1/pic_size),(ml_size,pic_size))
b1=np.random.normal(0,np.sqrt(1/pic_size),ml_size)
W2=np.random.normal(0,np.sqrt(1/ml_size),(class_num,ml_size))
b2=np.random.normal(0,np.sqrt(1/ml_size),class_num)

#作成済みプログラムを用いたニューラルネットワークの計算
_,X0=pc.choice(X)
flat_X0=il.flat(X0)
union_X0=ul.union(W1,flat_X0,b1)
y=sig.sigmoid(union_X0)
a=ul.union(W2,y,b2)
soft_a=sm.softmax(a).tolist()
result=dec.decision(soft_a)
print(result)
