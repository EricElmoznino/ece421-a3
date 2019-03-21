import tensorflow as tf


def distance_func(x, mu):
    # Inputs
    # x: is an NxD matrix (N observations and D dimensions)
    # mu: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    x = tf.reshape(x, [-1, 1, x.shape[1]])
    mu = tf.reshape(mu, [1, mu.shape[0], mu.shape[1]])
    pair_dist = tf.reduce_sum(((x - mu) ** 2), axis=-1)
    return pair_dist


def build_graph(d, k, lr):
    x = tf.placeholder(dtype=tf.float32, shape=[None, d])
    mu = tf.Variable(initial_value=tf.random_normal(shape=[k, d]))
    loss = distance_func(x, mu)
    assignments = tf.argmin(loss, axis=-1)
    loss = tf.reduce_sum(tf.reduce_min(loss, axis=-1))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.99, epsilon=1e-5)
    optimizer_op = optimizer.minimize(loss)
    return x, mu, assignments, loss, optimizer_op
