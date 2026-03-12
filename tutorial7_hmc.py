import numpy as np
from scipy.stats import norm
import os

def normal_pdf(x, mu, sigma):
    """PDF of normal distribution."""
    return norm.pdf(x, loc=mu, scale=sigma)

def neg_log_prob(x, mu, sigma):
    """
    Negative log probability (Potential Energy U) of N(mu, sigma).
    U(q) = (q - mu)^2 / (2 * sigma^2)
    """
    return (x - mu)**2 / (2 * sigma**2)

def grad_U(q, mu, sigma):
    """Gradient of negative log probability U(q) with respect to q."""
    return (q - mu) / sigma**2

def hmc_sampler(mu, sigma, step_size, num_steps, num_samples, initial_q):
    """
    Hamiltonian Monte Carlo sampler for a 1D normal distribution.
    
    Args:
        mu (float): Mean of the target normal distribution.
        sigma (float): Standard deviation of the target normal distribution.
        step_size (float): Leapfrog step size (epsilon).
        num_steps (int): Number of leapfrog steps (L).
        num_samples (int): Total number of samples to generate.
        initial_q (float): Initial position.
        
    Returns:
        samples (ndarray): Generated HMC samples.
        acceptance_rate (float): The fraction of proposed states that were accepted.
    """
    samples = np.zeros(num_samples)
    q = initial_q
    accepted = 0
    
    for i in range(num_samples):
        # 1. Sample initial momentum from standard normal distribution
        p = np.random.normal(0, 1)
        current_p = p
        current_q = q
        
        # 2. Leapfrog integration
        # Half step for momentum
        p = p - step_size * grad_U(q, mu, sigma) / 2.0
        
        # Alternate full steps for position and momentum
        for j in range(num_steps):
            # Full step for position
            q = q + step_size * p
            
            # Full step for momentum (except at the very last step)
            if j != num_steps - 1:
                p = p - step_size * grad_U(q, mu, sigma)
                
        # Final half step for momentum
        p = p - step_size * grad_U(q, mu, sigma) / 2.0
        
        # 3. Reverse momentum for time-reversibility condition 
        p = -p
        
        # 4. Evaluate potential (U) and kinetic (K) energies
        current_U = neg_log_prob(current_q, mu, sigma)
        current_K = current_p**2 / 2.0
        
        proposed_U = neg_log_prob(q, mu, sigma)
        proposed_K = p**2 / 2.0
        
        # 5. Metropolis acceptance step
        H_diff = current_U + current_K - proposed_U - proposed_K
        
        if H_diff >= 0:
            alpha = 1.0
        else:
            alpha = np.exp(H_diff)
            
        unif_rand = np.random.uniform(0, 1)
        if unif_rand < alpha:
            # Accept
            samples[i] = q
            accepted += 1
        else:
            # Reject: keep current state
            samples[i] = current_q
            q = current_q
            
    acceptance_rate = accepted / num_samples
    return samples, acceptance_rate

def run_experiment(step_size, num_steps, num_samples, mu, sigma, suffix):
    """Helper to run the sampler, print statistics, and save samples."""
    np.random.seed(42)
    initial_q = 0.0
    
    print(f"--- Experiment: Step Size (epsilon) = {step_size} ---")
    burn_in = int(0.1 * num_samples)
    total_samples_to_run = num_samples + burn_in
    
    raw_samples, acc_rate = hmc_sampler(mu, sigma, step_size, num_steps, total_samples_to_run, initial_q)
    
    samples = raw_samples[burn_in:]
    
    print(f"Acceptance Rate (overall): {acc_rate*100:.2f}%")
    print(f"Sample Mean:     {np.mean(samples):.4f}  (True value: {mu})")
    print(f"Sample Variance: {np.var(samples):.4f}  (True value: {sigma**2})")
    print()

    # Save to file
    os.makedirs('data', exist_ok=True)
    np.save(f'data/samples_{suffix}.npy', samples)
    
    return acc_rate, np.mean(samples), np.var(samples)

if __name__ == "__main__":
    print("Starting Tutorial 7 - HMC Experiments...\n")
    
    mu = 0.0
    sigma = 1.0
    num_steps = 20       
    num_samples = 5000   
    
    # Run experiments and export data
    run_experiment(0.25, num_steps, num_samples, mu, sigma, "025")
    run_experiment(0.8, num_steps, num_samples, mu, sigma, "08")
    
    print("Data successfully saved to the 'data/' directory. Run `generate_figures.py` to plot the results.")
