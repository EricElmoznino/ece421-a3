import tensorflow as tf
import numpy as np
import math as math
import helper as hlp


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
    return log_gauss


def log_posterior(log_pdf, log_pi):
    # Input
    # log_pdf: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    anormalized = log_pdf + tf.reshape(log_pi, [-1])
    return anormalized - hlp.reduce_logsumexp(anormalized, reduction_indices=1, keep_dims = True)


def neg_log_prob(x, mu, sigma, log_pi):
    log_pdf = log_gauss_pdf(x, mu, sigma) #pdf
    log_likelihood = hlp.reduce_logsumexp(log_pdf + tf.reshape(log_pi, [-1]))
    return -1 * log_likelihood


def build_graph(d, k, lr):
    x = tf.placeholder(dtype=tf.float32, shape=[None, d])
    mu = tf.Variable(initial_value=tf.random_normal(shape=[k, d]))
    sigma = tf.Variable(initial_value=tf.random_normal(shape=[k, 1]))
    pi = tf.Variable(initial_value=tf.random_normal(shape=[k, 1]))
    e_sigma = tf.exp(sigma)
    log_pi = hlp.logsoftmax(tf.reshape(pi,[-1]))
    loss = neg_log_prob(x, mu, e_sigma, log_pi)
    log_pdf = log_gauss_pdf(x, mu, e_sigma)
    log_post = log_posterior(log_pdf, log_pi)
    assignments = tf.argmax(log_post, axis=1)
    loss = tf.reduce_sum(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99, epsilon=1e-5)
    optimizer_op = optimizer.minimize(loss)
    return x, mu, sigma, pi, assignments, loss, optimizer_op