import mnist
import numpy as np

#以下，作成したファイルからのimport
import pic_choice as pc
import input_layer as il
import sigmoid as sig
import ramp
import dropout as dp
import union_layer as ul
import softmax as sm
import decision as dec

#MNIST画像の読み込み
X=mnist.download_and_parse_mnist_file("C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\t10k-images-idx3-ubyte.gz")
Y=mnist.download_and_parse_mnist_file("C:\\Users\\yamashita kouki\\TensorFlow-MNIST\\mnist\\data\\t10k-labels-idx1-ubyte.gz")
X=X/255

pic_size=28*28
pic_num=len(X)
class_num=10
ml_size=100
rho=0.1

#重みをファイルから読み込み
W1=np.load("W1_issue3.npy")
b1=np.load("b1_issue3.npy")
W2=np.load("W2_issue3.npy")
b2=np.load("b2_issue3.npy")

#テスト画像をまとめて与え，正答率を表示
X_list=(X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))).T
Y1=ul.union(W1,X_list,b1)
X2_list=sig.sigmoid(Y1)
Y2=ul.union(W2,X2_list,b2)
results=sm.softmax(Y2)
result=np.argmax(results,axis=0)
print("correct(sigmoid): " + str(np.count_nonzero(result==Y)) + "/" + str(pic_num))


#重みをファイルから読み込み
W1=np.load("W1_issueA1.npy")
b1=np.load("b1_issueA1.npy")
W2=np.load("W2_issueA1.npy")
b2=np.load("b2_issueA1.npy")

#テスト画像をまとめて与え，正答率を表示
X_list=(X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))).T
Y1=ul.union(W1,X_list,b1)
X2_list=ramp.ReLU(Y1)
Y2=ul.union(W2,X2_list,b2)
results=sm.softmax(Y2)
result=np.argmax(results,axis=0)
print("correct(ReLU): " + str(np.count_nonzero(result==Y)) + "/" + str(pic_num))


#重みをファイルから読み込み
W1=np.load("W1_issueA2.npy")
b1=np.load("b1_issueA2.npy")
W2=np.load("W2_issueA2.npy")
b2=np.load("b2_issueA2.npy")

#テスト画像をまとめて与え，正答率を表示
X_list=(X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))).T
Y1=ul.union(W1,X_list,b1)
X2_list=dp.drop(Y1,rho,istest=True)
Y2=ul.union(W2,X2_list,b2)
results=sm.softmax(Y2)
result=np.argmax(results,axis=0)
print("correct(Dropout): " + str(np.count_nonzero(result==Y)) + "/" + str(pic_num))

#入力された番号の画像の識別
while(True):
    num, X0=pc.choice(X)
    flat_X0=il.flat(X0).reshape(pic_size,1)
    union_X0=ul.union(W1,flat_X0,b1)
    y1=dp.drop(union_X0,rho,istest=True)
    a=ul.union(W2,y1,b2)
    soft_a=sm.softmax(a).tolist()
    result=dec.decision(soft_a)
    print("No." + str(num) + " is " + str(Y[num]) + ", expected " + str(result))
