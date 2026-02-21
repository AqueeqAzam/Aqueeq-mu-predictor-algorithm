"""Visualization functions for μ-Predictor."""

import matplotlib.pyplot as plt
import numpy as np
from .core import MuPredictor

def plot_psi_series(predictor, depth=100):
    """Plot ψ(k) series to visualize convergence."""
    psi = predictor.compute_psi_series(depth)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(depth+1), psi, 'b-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='ψ → 1')
    plt.xlabel('k')
    plt.ylabel('ψ(k)')
    plt.title(f'ψ(k) Convergence (μ={predictor.mu:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_stability_map(a_range, b_range, c=1.0, d=1.0, points=50):
    """Plot stability map in a-b parameter space."""
    a_vals = np.linspace(a_range[0], a_range[1], points)
    b_vals = np.linspace(b_range[0], b_range[1], points)
    A, B = np.meshgrid(a_vals, b_vals)
    MU = B * A**(-c * d / 2)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(A, B, MU, levels=20, cmap='RdYlGn_r')
    plt.colorbar(label='μ')
    plt.contour(A, B, MU, levels=[1], colors='black', linewidths=2)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title('Stability Map (μ < 1 = Green, μ > 1 = Red)')
    plt.show()
