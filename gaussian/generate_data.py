import numpy as np
import matplotlib.pyplot as plt


class TwoGaussian(object):

    def __init__(self, **kwargs):
        self.mu1 = np.array([0, 0])
        self.mu2 = np.array([3, 3.5])
        self.cov = np.array([[1, 0], [0, 1]])
        self.positive_prior = 0.5
        self.positive_samples = []
        self.negative_samples = []
        self.unlabeled_samples = []
        self.observed_negative_samples = []
        self.__dict__.update(kwargs)

    def positive_posterior(self, x):
        conditional_positive = (
            np.exp(-0.5*(x-self.mu1).T.dot(x-self.mu1)) / (2*np.pi))
        conditional_negative = (
            np.exp(-0.5*(x-self.mu2).T.dot(x-self.mu2)) / (2*np.pi))
        marginal_dist = (
            self.positive_prior * conditional_positive
            + (1-self.positive_prior) * conditional_negative)
        positive_posterior = (
            conditional_positive * self.positive_prior / marginal_dist)
        return positive_posterior

    # x is with label -1
    def neg_observed_prob(self, x):
        xp = self.positive_posterior(x)
        ob_prob = 0 if xp < 1e-6 else xp**(1/10)
        return ob_prob

    def observed_prob(self, x):
        xp_prob = self.positive_posterior(x)
        xon_prob = self.neg_observed_prob(x) * (1-xp_prob)
        return xp_prob + xon_prob

    def estimate_neg_observed_prob(self, n):
        n_samples = self.draw_negative(n, store=False)
        ob_probs = []
        for i in range(n_samples.shape[0]):
            x = n_samples[i]
            ob_probs.append(self.neg_observed_prob(x))
        return (np.mean(np.array(ob_probs)) * (1-self.positive_prior)).item()

    def draw_positive(self, n, store=True):
        drawn = np.random.multivariate_normal(self.mu1, self.cov, n)
        if store:
            self.positive_samples.extend(drawn)
        return drawn

    def draw_negative(self, n, store=True):
        drawn = np.random.multivariate_normal(self.mu2, self.cov, n)
        if store:
            self.negative_samples.extend(drawn)
        return drawn

    def draw_unlabeled(self, n, store=True):
        n_positive = np.random.binomial(n, self.positive_prior)
        positive_samples = self.draw_positive(n_positive, store=False)
        negative_samples = self.draw_negative(n-n_positive, store=False)
        u_samples = np.r_[positive_samples, negative_samples]
        self.unlabeled_samples.extend(u_samples)
        return np.random.permutation(u_samples)

    def draw_observed_negative(self, n, store=True):
        observed = []
        while len(observed) < n:
            x = self.draw_negative(1, store=False)[0]
            if np.random.random() < self.neg_observed_prob(x):
                observed.append(x)
        if store:
            self.observed_negative_samples.extend(observed)
        return np.array(observed)

    def plot_samples(self, **kwargs):
        if self.unlabeled_samples != []:
            ux, uy = np.array(self.unlabeled_samples).T
            plt.scatter(ux, uy, color='greenyellow', s=3, alpha=0.5,
                        label='unlabeled', **kwargs)
        if self.positive_samples != []:
            px, py = np.array(self.positive_samples).T
            plt.scatter(px, py, color='salmon', s=3, alpha=0.5,
                        label='positive', **kwargs)
        if self.negative_samples != []:
            nx, ny = np.array(self.negative_samples).T
            plt.scatter(nx, ny, color='turquoise', s=3, alpha=0.5,
                        label='negative', **kwargs)
        if self.observed_negative_samples != []:
            onx, ony = np.array(self.observed_negative_samples).T
            plt.scatter(onx, ony, color='navy', s=3, alpha=0.5,
                        label='observed negative', **kwargs)

    def clear_samples(self):
        self.positive_samples = []
        self.negative_samples = []
        self.unlabeled_samples = []
        self.observed_negative_samples = []


class ThreeGaussian(object):

    def __init__(self, **kwargs):
        self.mu1 = np.array([0, 0])
        self.mu2 = np.array([1.5, 2])
        self.mu3 = np.array([-5, -5])
        self.cov = np.array([[1, 0], [0, 1]])
        self.on1_prob = 0.8
        self.positive_samples = []
        self.negative_samples = []
        self.unlabeled_samples = []
        self.observed_negative_samples = []
        self.__dict__.update(kwargs)

    def observed_prob(self, x):
        conditional_positive = (
            np.exp(-0.5*(x-self.mu1).T.dot(x-self.mu1)) / (2*np.pi))
        conditional_negative1 = (
            np.exp(-0.5*(x-self.mu2).T.dot(x-self.mu2)) / (2*np.pi))
        conditional_negative2 = (
            np.exp(-0.5*(x-self.mu3).T.dot(x-self.mu3)) / (2*np.pi))
        marginal_dist = (
            1/2 * conditional_positive
            + 1/4 * conditional_negative1
            + 1/4 * conditional_negative2)
        ob_prob = ((
            1/2 * conditional_positive
            + 1/4 * self.on1_prob * conditional_negative1) / marginal_dist)
        return ob_prob

    def estimate_neg_observed_prob(self, n):
        return 1/4 * self.on1_prob

    def draw_positive(self, n, store=True):
        drawn = np.random.multivariate_normal(self.mu1, self.cov, n)
        if store:
            self.positive_samples.extend(drawn)
        return drawn

    def draw_negative(self, n, store=True):
        n_negative1 = np.random.binomial(n, 1/2)
        n1_samples = np.random.multivariate_normal(
            self.mu2, self.cov, n_negative1)
        n2_samples = np.random.multivariate_normal(
            self.mu3, self.cov, n-n_negative1)
        drawn = np.r_[n1_samples, n2_samples]
        if store:
            self.negative_samples.extend(drawn)
        return drawn

    def draw_unlabeled(self, n, store=True):
        n_positive = np.random.binomial(n, 1/2)
        positive_samples = self.draw_positive(n_positive, store=False)
        negative_samples = self.draw_negative(n-n_positive, store=False)
        u_samples = np.r_[positive_samples, negative_samples]
        self.unlabeled_samples.extend(u_samples)
        return np.random.permutation(u_samples)

    def draw_observed_negative(self, n, store=True):
        drawn = np.random.multivariate_normal(self.mu2, self.cov, n)
        if store:
            self.observed_negative_samples.extend(drawn)
        return drawn

    def plot_samples(self, **kwargs):
        if self.unlabeled_samples != []:
            ux, uy = np.array(self.unlabeled_samples).T
            plt.scatter(ux, uy, color='greenyellow', s=3, alpha=0.5,
                        label='unlabeled', **kwargs)
        if self.positive_samples != []:
            px, py = np.array(self.positive_samples).T
            plt.scatter(px, py, color='salmon', s=3, alpha=0.5,
                        label='positive', **kwargs)
        if self.negative_samples != []:
            nx, ny = np.array(self.negative_samples).T
            plt.scatter(nx, ny, color='turquoise', s=3, alpha=0.5,
                        label='negative', **kwargs)
        if self.observed_negative_samples != []:
            onx, ony = np.array(self.observed_negative_samples).T
            plt.scatter(onx, ony, color='navy', s=3, alpha=0.5,
                        label='observed negative', **kwargs)

    def clear_samples(self):
        self.positive_samples = []
        self.negative_samples = []
        self.unlabeled_samples = []
        self.observed_negative_samples = []
