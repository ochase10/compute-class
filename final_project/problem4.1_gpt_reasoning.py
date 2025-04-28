import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Define the target density p(x): Gaussian with mean=2, variance=2
def p(x):
    return 1/np.sqrt(2 * np.pi * 2) * np.exp(-((x - 2) ** 2) / (2 * 2))

# Metropolis-Hastings sampler parameters
n_steps = 20000  # > 10^4
samples = np.zeros(n_steps)
x = 0.0  # Initial state

# Run the sampler
for i in range(n_steps):
    x_prop = np.random.normal(loc=x, scale=1.0)  # Proposal q(x'|x) ~ N(x, 1)
    # Acceptance probability
    alpha = p(x_prop) / p(x)
    if np.random.rand() < alpha:
        x = x_prop
    samples[i] = x

# Discard a burn-in period
burn_in = 1000
samples_post = samples[burn_in:]

# Plotting
plt.figure()
plt.hist(samples_post, bins=50, density=True)
x_line = np.linspace(min(samples_post) - 1, max(samples_post) + 1, 300)
plt.plot(x_line, p(x_line), linewidth=2)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Metropolis-Hastings Samples vs True Density')
plt.show()
