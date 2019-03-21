import tensorflow as tf
import numpy as np
import math as math
import matplotlib.pyplot as plt
import helper as hlp


# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]


# Distance function for GMM
def distance_func(x, mu):
    # Inputs
    # x: is an NxD matrix (N observations and D dimensions)
    # mu: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # todo
    pass


def log_gauss_pdf(x, mu, sigma):
    # Inputs
    # x: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    x = tf.reshape(x, [x.shape[0], 1, x.shape[1]])
    mu = tf.reshape(mu, [1, mu.shape[0], mu.shape[1]])
    log_gauss = -1 * tf.reduce_sum(((x - mu) ** 2), axis=-1) / (2 * sigma ** 2) - tf.math.log(
        2 * tf.math.pi ** (mu.shape[1]/2) * sigma)
    #log_gauss = -1 * tf.reduce_sum(((x - mu) ** 2), axis=-1) / (2 * sigma ** 2) - tf.math.log(
        #2 * tf.math.pi ** (mu.shape[1] / 2) * sigma ** 2) / 2
    return log_gauss


def log_posterior(log_pdf):
    # Input
    # log_pdf: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    return log_pdf - hlp.logsoftmax(log_pdf)
