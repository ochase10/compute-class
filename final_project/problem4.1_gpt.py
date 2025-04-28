import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Target density: Gaussian with mean=2, variance=2
def p(x):
    return norm.pdf(x, loc=2, scale=np.sqrt(2))

# Proposal distribution: Gaussian centered at current x, variance=1
def q(x_prime, x):
    return norm.pdf(x_prime, loc=x, scale=1)

def metropolis_hastings(p, q, q_sample, initial_x, n_steps):
    x = initial_x
    samples = [x]
    
    for _ in range(n_steps):
        x_prime = q_sample(x)  # propose new x'
        acceptance_ratio = p(x_prime) * q(x, x_prime) / (p(x) * q(x_prime, x))
        acceptance_ratio = min(1, acceptance_ratio)
        
        if np.random.rand() < acceptance_ratio:
            x = x_prime
        
        samples.append(x)
    
    return np.array(samples)

# q_sample: sample from proposal distribution
def q_sample(x):
    return np.random.normal(loc=x, scale=1)

# Settings
initial_x = 0
n_steps = 12000  # More than 10^4

# Run sampler
samples = metropolis_hastings(p, q, q_sample, initial_x, n_steps)

# Plot results
x_plot = np.linspace(-2, 6, 500)
true_density = norm.pdf(x_plot, loc=2, scale=np.sqrt(2))

plt.figure(figsize=(10,6))
plt.hist(samples, bins=50, density=True, alpha=0.6, label='MCMC samples')
plt.plot(x_plot, true_density, 'r-', lw=2, label='True density')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Metropolis-Hastings Sampling')
plt.legend()
plt.show()
