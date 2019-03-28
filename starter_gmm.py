import tensorflow as tf
import numpy as np
import math as math
import matplotlib.pyplot as plt
import helper as hlp


# Loading data
#data = np.load('data100D.npy')
"""
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

"""
# Distance function for GMM
def distance_func(x, mu):
    # Inputs
    # x: is an NxD matrix (N observations and D dimensions)
    # mu: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    x = tf.reshape(x, [-1, 1, int(x.shape[1])])
    mu = tf.reshape(mu, [1, int(mu.shape[0]), int(mu.shape[1])])
    pair_dist = tf.reduce_sum(((x - mu) ** 2), axis=-1)
    return pair_dist


def log_gauss_pdf(x, mu, sigma):
    # Inputs
    # x: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    pair_dist = distance_func(x, mu)
    log_gauss = -1 * pair_dist/ (2 * tf.reshape(sigma ** 2, [-1])) - int(mu.shape[1]) * tf.log(np.sqrt(2 * math.pi) * tf.reshape(sigma, [-1]))
    #log_gauss = -1 * tf.reduce_sum(((x - mu) ** 2), axis=-1) / (2 * sigma ** 2) - tf.math.log(
        #2 * tf.math.pi ** (mu.shape[1] / 2) * sigma ** 2) / 2
    return log_gauss


def log_posterior(log_pdf, log_pi):
    # Input
    # log_pdf: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    return log_pdf + tf.reshape(log_pi, [-1]) - hlp.reduce_logsumexp(log_pdf + tf.reshape(log_pi, [-1]), reduction_indices=1, keep_dims = True) #iog_pdf- reduce_logsumexp(log_pdf, reduction_indices=0, keep_dims=True)

def neg_log_prob(x, mu, sigma, log_pi):
    log_pdf = log_gauss_pdf(x, mu, sigma) #pdf
    log_likelihood = tf.reshape(hlp.reduce_logsumexp(log_pdf+tf.reshape(log_pi, [-1])), [-1, 1])
    return -1 * log_likelihood

def build_graph(d, k, lr):
    x = tf.placeholder(dtype=tf.float32, shape=[None, d])
    mu = tf.Variable(initial_value=tf.random_normal(shape=[k, d]))
    sigma = tf.Variable(initial_value=tf.random_normal(shape=[k, 1]))
    pi = tf.Variable(initial_value=tf.random_normal(shape=[k, 1]))
    log_pi = hlp.logsoftmax(tf.reshape(pi,[-1]))
    loss = neg_log_prob(x, mu, tf.exp(sigma), log_pi)
    assignments = tf.argmin(distance_func(x, mu), axis=-1)
    loss = tf.reduce_sum(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99, epsilon=1e-5)
    optimizer_op = optimizer.minimize(loss)
    return x, mu, sigma, pi, assignments, loss, optimizer_op