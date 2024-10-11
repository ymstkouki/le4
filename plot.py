import matplotlib.pyplot as plt
import numpy as np

en_sig=np.load("entropy_sigmoid.npy")
en_ReLU=np.load("entropy_ReLU.npy")
en_drop_sig=np.load("entropy_dropout_sigmoid.npy")
en_drop_ReLU=np.load("entropy_dropout_ReLU.npy")

plt.plot(list(range(len(en_sig))),en_sig,color="r",label="sigmoid")
plt.plot(list(range(len(en_ReLU))),en_ReLU,color="g",label="ReLU")
plt.plot(list(range(len(en_drop_sig))),en_drop_sig,color="b",label="sigmoid&dropout")
plt.plot(list(range(len(en_drop_ReLU))),en_drop_ReLU,color="y",label="ReLU&dropout")

plt.legend()

plt.show()
