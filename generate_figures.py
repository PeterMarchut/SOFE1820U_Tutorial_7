import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import os

def normal_pdf(x, mu, sigma):
    """PDF of normal distribution."""
    return norm.pdf(x, loc=mu, scale=sigma)

def plot_histogram(samples, step_size, mu, sigma, title, filename):
    """Plot histogram of samples compared to true density."""
    plt.figure(figsize=(8, 5))
    
    # Plot histogram of samples
    count, bins, ignored = plt.hist(samples, bins=50, density=True, alpha=0.6, 
                                    color='#1f77b4', edgecolor='black', label=f'HMC Samples ($\\epsilon$={step_size})')
    
    # Plot true target density
    x_val = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    plt.plot(x_val, normal_pdf(x_val, mu, sigma), 'r-', lw=2.5, label='True Density $N(0, 1)$')
    
    plt.title(title, fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_2d_demo():
    """
    Optional: Visualizes a strong 2D correlated Gaussian to explain 
    why Metropolis-Hastings random walk struggles compared to HMC.
    """
    # Create a 2D highly correlated target distribution
    mean = [0, 0]
    cov = [[1.0, 0.95], [0.95, 1.0]]
    rv = multivariate_normal(mean, cov)
    
    # Grid for contour plot
    x, y = np.mgrid[-3:3:.05, -3:3:.05]
    pos = np.dstack((x, y))
    
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, rv.pdf(pos), levels=15, cmap='viridis')
    
    # Annotate MH random walk behavior (slow to traverse narrow regions)
    plt.annotate(
        "Random Walk (MH) Proposals\n(often rejected or move slowly)",
        xy=(1.5, 1.5), xytext=(2, -1),
        arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
        color='white', fontweight='bold', ha='center'
    )
    
    # Annotate HMC trajectory behavior (smooth integration along contours)
    plt.annotate(
        "HMC Trajectory\n(moves efficiently along density ridges)",
        xy=(-1.5, -1.5), xytext=(-2.5, 1.5),
        arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8),
        color='white', fontweight='bold', ha='center'
    )
    
    plt.title("2D Correlated Gaussian: MH vs. HMC Exploration", fontsize=14)
    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$x_2$", fontsize=12)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/optional_2d_demo.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating figures from saved data...\n")
    mu = 0.0
    sigma = 1.0
    
    # Question 2: Experiment with step_size = 0.25
    if os.path.exists('data/samples_025.npy'):
        samples_025 = np.load('data/samples_025.npy')
        plot_histogram(samples_025, 0.25, mu, sigma, 
                       '1D HMC Sampling with Step Size $\\epsilon = 0.25$', 
                       'figures/eps_025_hist.png')
        print("Generated figures/eps_025_hist.png")
    else:
        print("Warning: data/samples_025.npy not found. Please run tutorial7_hmc.py first.")

    # Question 3: Experiment with step_size = 0.8
    if os.path.exists('data/samples_08.npy'):
        samples_08 = np.load('data/samples_08.npy')
        plot_histogram(samples_08, 0.8, mu, sigma, 
                       '1D HMC Sampling with Step Size $\\epsilon = 0.8$', 
                       'figures/eps_08_hist.png')
        print("Generated figures/eps_08_hist.png")
    else:
        print("Warning: data/samples_08.npy not found. Please run tutorial7_hmc.py first.")
        
    # Question 4: Optional 2D visualization
    plot_2d_demo()
    print("Generated figures/optional_2d_demo.png")
    
    print("\nAll figures successfully generated in the 'figures/' directory.")
