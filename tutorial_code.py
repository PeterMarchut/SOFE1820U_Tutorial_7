import numpy as np
import random
import scipy.stats as st
import matplotlib.pyplot as plt


def normal(x, mu, sigma):
    numerator = np.exp(-1 * ((x - mu) ** 2) / (2 * sigma ** 2))
    denominator = sigma * np.sqrt(2 * np.pi)
    return numerator / denominator


def neg_log_prob(x, mu, sigma):
    return -1 * np.log(normal(x=x, mu=mu, sigma=sigma))


def HMC(mu=0.0, sigma=1.0, path_len=1, step_size=0.25, initial_position=0.0, epochs=1000):

    # setup
    steps = int(path_len / step_size)
    samples = [initial_position]
    momentum_dist = st.norm(0, 1)

    # generate samples
    for e in range(epochs):

        q0 = np.copy(samples[-1])
        q1 = np.copy(q0)

        p0 = momentum_dist.rvs()
        p1 = np.copy(p0)

        # gradient of PDF wrt position
        dVdQ = -1 * (q0 - mu) / (sigma ** 2)

        # leapfrog integration begin
        for s in range(steps):

            # half step momentum update
            p1 += step_size * dVdQ / 2

            # position update
            q1 += step_size * p1

            # second half step momentum update
            p1 += step_size * dVdQ / 2

        # leapfrog integration end

        # flip momentum for reversibility
        p1 = -1 * p1

        # metropolis acceptance
        q0_nlp = neg_log_prob(x=q0, mu=mu, sigma=sigma)
        q1_nlp = neg_log_prob(x=q1, mu=mu, sigma=sigma)

        p0_nlp = neg_log_prob(x=p0, mu=0, sigma=1)
        p1_nlp = neg_log_prob(x=p1, mu=0, sigma=1)

        # acceptance probability
        target = q0_nlp - q1_nlp
        adjustment = p1_nlp - p0_nlp
        acceptance = target + adjustment

        event = np.log(random.uniform(0, 1))

        if event <= acceptance:
            samples.append(q1)
        else:
            samples.append(q0)

    return samples


# run experiment
mu = 0
sigma = 1

trial = HMC(mu=mu, sigma=sigma, path_len=1.5, step_size=0.25)

lines = np.linspace(-6, 6, 10000)
normal_curve = [normal(x=l, mu=mu, sigma=sigma) for l in lines]

plt.plot(lines, normal_curve)
plt.hist(trial, density=True, bins=20)

plt.show()