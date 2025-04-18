import scipy.stats as sts
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the range of mu
mu = np.linspace(1.65, 1.8, num=50)

# Generate the uniform distribution
uniform_dist = sts.uniform.pdf(mu) + 1
uniform_dist = uniform_dist / uniform_dist.sum()  # Normalize the distribution

# Generate the beta distribution
beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.2)
beta_dist = beta_dist / beta_dist.sum()  # Normalize the distribution

# Define the likelihood function
def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)  # Note that mu here is an array of values, so the output is also an array!
    return likelihood_out / likelihood_out.sum()  # Normalize the likelihoods

# Compute the likelihoods for the observed datum of 1.7m
likelihood_out = likelihood_func(1.7, mu)



unnormalized_posterior = likelihood_out * uniform_dist


p_data = sp.integrate.trapz(unnormalized_posterior, mu)
normalized_posterior = unnormalized_posterior/(p_data)

heights_data = sts.norm.rvs(loc = 1.7, scale = 0.1, size = 1001)

prior = uniform_dist
posterior_dict = {}

# Create subplots
fig, axs = plt.subplots(1,2, figsize=(14, 5))

# Plot the uniform and beta distributions in the first subplot
axs[0].plot(mu, beta_dist, label='Beta Distribution')
axs[0].plot(mu, uniform_dist, label='Uniform Distribution')
axs[0].set_xlabel("Value of $\mu$ in meters")
axs[0].set_ylabel("Probability Density")
axs[0].set_title("Uniform and Beta Distributions")
axs[0].legend()

# Plot the likelihood in the second subplot
axs[1].plot(mu, likelihood_out)
axs[1].set_title("Likelihood of $\mu$ given observation 1.7m")
axs[1].set_ylabel("Probability Density/Likelihood")
axs[1].set_xlabel("Value of $\mu$")

fig, axs = plt.subplots(1,2, figsize=(14, 5))

axs[0].plot(mu, unnormalized_posterior)
axs[0].set_xlabel("$\mu$ in meters")
axs[0].set_ylabel("Unnormalized Posterior")

axs[1].plot(mu, normalized_posterior)
axs[1].set_xlabel("$\mu$ in meters")
axs[1].set_ylabel("Probability Density")




plt.figure(figsize = (10, 8))

for ind, datum in enumerate(heights_data):
  likelihood = likelihood_func(datum, mu)
  unnormalized_posterior = prior * likelihood
  normalized_posterior = unnormalized_posterior/sp.integrate.trapz(unnormalized_posterior, mu)
  prior = normalized_posterior
  posterior_dict[ind] = normalized_posterior
  if ind%200 == 0:
    plt.plot(mu, normalized_posterior, label = f'Model after observing {ind} data')

plt.legend()
plt.show()
