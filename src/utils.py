import numpy as np
from numpy.random import normal, randint, uniform
from functools import reduce


def generate_training_data(number_gaussian_funcs, max_mean=100, max_std=1):
    """
    Generate training data resulting of mixing a number of gaussian distributions
    together
    :param number_gaussian_funcs: The number of functions to mix in
    :param max_mean: the maximum mean a distribution can have, mean will be in range [-max_mean, max_mean]
    :param max_std: the maximum standard deviation the gaussian functions can have [0, max_std]
    :return:
    """
    # Generate the parameters for these gaussian functions
    gauss_params = [(uniform(-max_mean, max_mean), uniform(0, max_std)) for i in range(number_gaussian_funcs)]
    # Number of data points per gaussian distribution
    num_pts = 50

    # Generate the training data for each gaussian distribution
    distributions = [np.array([normal(mean, std) for _ in range(num_pts)]) for (mean, std) in gauss_params]

    # Merge all the distributions together
    merged = distributions[0]
    for dist in distributions[1:]:
        merged += dist

    return merged


def gaussian(x, mean, standard_deviation):
    """
    This is the probability density function of a normal distribution
    STANDARD - Stormzy
    :param mean:
    :param standard_deviation:
    :param x:
    :return:
    """
    # Avoid division by zero by adding a small number
    if standard_deviation == 0:
        standard_deviation += 1e-10

    return np.e ** (-(x - mean) ** 2 / (2 * standard_deviation ** 2)) / (standard_deviation * (2 * np.pi) ** 0.5)


def predict_point_probabilities(training_data, param_pairs):
    """
    Calculate the probability of each point in x belonging to a specific cluster
    :param training_data:
    :param param_pairs
    :return:
    """
    return [np.argmax([gaussian(x, mean, std) for mean, std in param_pairs]) for x in training_data]