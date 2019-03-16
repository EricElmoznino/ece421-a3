import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


# Distance function for K-means
def distance_func(x, mu):
    # Inputs
    # x: is an NxD matrix (N observations and D dimensions)
    # mu: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    x = tf.reshape(x, [x.shape[0], 1, x.shape[1]])
    mu = tf.reshape(mu, [1, mu.shape[0], mu.shape[1]])
    pair_dist = tf.reduce_sum(((x - mu) ** 2), axis=-1)
    return pair_dist
