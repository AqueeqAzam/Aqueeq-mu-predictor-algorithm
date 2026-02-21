"""Utility functions for μ-Predictor."""

import numpy as np

def mu_from_parameters(a, b, c=1.0, d=1.0):
    """Calculate μ directly from parameters."""
    return b * a**(-c * d / 2)

def classify_mu(mu):
    """Classify μ value."""
    if mu < 1:
        return "CONVERGENT"
    elif mu > 1:
        return "DIVERGENT"
    else:
        return "CRITICAL"

def find_critical_a(b, c=1.0, d=1.0):
    """Find a such that μ = 1."""
    return (b)**(2/(c*d))
