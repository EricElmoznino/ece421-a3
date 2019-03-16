import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


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

    # todo
    pass


def log_posterior(log_pdf, log_pi):
    # Input
    # log_pdf: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # todo
    pass
