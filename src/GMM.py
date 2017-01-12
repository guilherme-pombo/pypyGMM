import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from numpy import arange
from utils import generate_training_data, gaussian, predict_point_probabilities


"""
This file has a Gaussian Mixture Model implementation that works for 1-D points
"""


class GMM(object):

    def __init__(self, number_components, max_iterations=100, bic_weight=10):
        """
        GMM class constructor
        :param number_components: how many gaussian functions or model attempts to find
        :param max_iterations: how many times to run EM for (unless likelihood stops improving)
        :param bic_weight: How much weight to give to lower model complexity, higher weight leads to choice
        of simpler mixture model
        """
        self.number_components = number_components
        self.max_iterations = max_iterations
        self.params = None
        self.likelihood = None
        self.bic = None
        self.bic_weight = bic_weight

    def em_algorithm(self, training_data):
        """
        Do the Expectation step followed by Maximisation step for the defined number of iterations
        The objective is to minimize the log likelihood and keeping track
        of the parameters that achieved the best results
        :param training_data:
        :return:
        """
        training_data = np.array(training_data)
        min = training_data.min()
        max = training_data.max()
        std = np.std(training_data)

        best_likelihood = float('-inf')
        best_params = None

        # Sample 100 pairs of a random number sampled from a uniform distribution with pdf p(x) = 1/max - min
        param_pairs = [(uniform(min, max), std) for _ in range(self.number_components)]
        previous_likelihood = None

        for epoch in range(self.max_iterations):

            # E-step
            cluster_ids = predict_point_probabilities(training_data, param_pairs)
            cluster_to_points = {c: [] for c in range(len(param_pairs))}

            # Set each point in the data to belong to a cluster
            for idx, cluster in enumerate(cluster_ids):
                cluster_to_points[cluster].append(idx)

            # M-step
            log_likelihood = 0
            for c, _ in enumerate(param_pairs):
                pts = training_data[cluster_to_points[c]]

                if len(pts) == 0:
                    continue
                else:
                    mean = np.mean(pts)
                    # Add smoothing parameter to avoid division by zero, for clusters with single point and std=0
                    std = np.std(pts) + 1e-10
                    param_pairs[c] = (mean, std)

                log_likelihood += sum(np.log(gaussian(x, mean, std)) for x in pts)

            # If log_likelihood doesn't change, stop iterations
            if log_likelihood == previous_likelihood:
                break

            previous_likelihood = log_likelihood

            # Keep track of the best likelihood and which parameters achieved this
            # Remember minimising log_likelihood is the same as maximising likelihood
            if log_likelihood > best_likelihood:
                best_likelihood = log_likelihood
                best_params = list(param_pairs)

        # calculate Bayesian Information Criterion
        self.params = best_params
        self.likelihood = best_likelihood
        # Penalise high number of components
        self.bic = self.bic_weight * self.number_components * np.log(training_data.shape[0]) - 2 * best_likelihood


def find_best_k(training_data, max_num_components=25):
    """
    Train various GMMs and return the k which achieves the lower Bayesian Information Criterion
    BIC is a criterion for model selection among a finite set of models;
    the model with the lowest BIC is preferred
    :param training_data: the training data
    :param max_num_components: the maximum number of components to try out
    :return:
    """
    best_bic = float('inf')
    best_k = None
    for k in range(1, max_num_components):
        clf = GMM(k)
        clf.em_algorithm(training_data)
        if clf.bic < best_bic:
            best_bic = clf.bic
            best_k = k

    return best_k


if __name__ == '__main__':
    # Generate training data
    number_gaussian_funcs = 5
    training_data = generate_training_data(number_gaussian_funcs, max_mean=100, max_std=10)

    # Find the best number of gaussian functions to use
    best_k = find_best_k(training_data)

    # Use this best number to train a new Gaussian Mixture Model
    clf = GMM(best_k)
    clf.em_algorithm(training_data)

    # Plot the various Gaussian functions that make up the mixture
    training_data = arange(min(training_data), max(training_data), (max(training_data) - min(training_data)) / 1000)
    for mean, std in clf.params:
        plt.plot(training_data, [gaussian(x, mean, std) for x in training_data])
    plt.show()

    print("Number of gaussian components in data: " + str(number_gaussian_funcs) +
          "\nPredicted number of gaussian components: " + str(best_k))
