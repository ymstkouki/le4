import numpy as np

#初期値：gamma=1, beta=0

def bnormalize(x, gamma, beta):
    epsilon = 1e-8

    batch_size = x.shape[-1]
    mu = np.sum(x, axis=1).reshape(-1, 1) / batch_size
    sigma = (x - mu) ** 2 / batch_size
    x_ = (x - mu) / np.sqrt(sigma + epsilon)
    y = gamma * x_ + beta

    return y, x_, mu, sigma

def bnorm_bp(en_y, x, gamma, x_, mu, sigma):
    epsilon = 1e-8
    batch_size = x.shape[-1]
    #print(f'x_ : {x_}')

    en_x_ = en_y * gamma
    #print(f'en_x_ : {en_x_.shape}')
    en_sigma = np.sum(en_x_ * (x - mu) * (-1/2) * ((sigma + epsilon) ** (-3/2)), axis=1).reshape(-1, 1)
    #print(f'en_sigma : {en_sigma.shape}')
    en_mu = -np.sum(en_x_ / np.sqrt(sigma + epsilon), axis=1).reshape(-1, 1) + en_sigma * np.sum(-2 * (x - mu) / batch_size, axis=1).reshape(-1, 1)
    #print(f'en_mu : {en_mu.shape}')
    en_x = en_x_ / np.sqrt(sigma + epsilon) + en_sigma * (2 * (x - mu) / batch_size) + en_mu / batch_size
    #print(f'en_x : {en_x.shape}')
    en_gamma = np.sum(en_y * x_, axis=1).reshape(-1, 1)
    #print(f'en_gamma : {en_gamma.shape}')
    en_beta = np.sum(en_y, axis=1).reshape(-1, 1)
    #print(f'en_beta : {en_beta}')

    return en_x, en_gamma, en_beta
