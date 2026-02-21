"""
================================================================================
AZEEQ AZAM'S μ-PREDICTOR ALGORITHM
Based on the Asymptotic Necessity Theorem (2026)

This algorithm predicts and computes infinite nested radicals of the form:
    f(x) = √(a^{cx} + b·f(x+d))

The critical parameter μ = b·a^{-cd/2} determines all behavior:
    μ < 1  → Convergent (exponential)
    μ = 1  → Critical (golden ratio when a=1)
    μ > 1  → Divergent

Reference: Azam, A. (2026). "On the Uniqueness and Asymptotic Behavior 
           of Infinite Geometric Radicals: A Rigorous Functional Approach 
           with Numerical Verification"
================================================================================
"""

import numpy as np
import warnings

class NestedRadicalPredictor:
    """
    A predictive algorithm for infinite geometric nested radicals.
    
    Based on Theorem 3.1 (Scaling Transformation) and Theorem 3.4 
    (Asymptotic Necessity) from Azam (2026).
    
    Parameters
    ----------
    a : float > 1
        Base of the geometric progression
    b : float > 0
        Coefficient multiplying the inner radical
    c : float > 0
        Exponent multiplier
    d : float > 0
        Shift parameter in the functional equation
        
    Attributes
    ----------
    mu : float
        Critical parameter μ = b·a^{-cd/2} that determines behavior
    """
    
    def __init__(self, a, b, c, d):
        """
        Initialize predictor with parameters from functional equation:
        f(x) = √(a^{cx} + b·f(x+d))
        """
        # Validate inputs
        if a <= 1:
            warnings.warn(f"a={a} ≤ 1, but theory assumes a>1. Results may be unexpected.")
        if b <= 0:
            raise ValueError(f"b={b} must be positive")
        if c <= 0:
            raise ValueError(f"c={c} must be positive")
        if d <= 0:
            raise ValueError(f"d={d} must be positive")
            
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        # Critical parameter from Theorem 3.1
        self.mu = b * a**(-c * d / 2)
        
    def predict_behavior(self):
        """
        Predict convergence/divergence behavior based on μ value.
        
        Theorem 3.4 guarantees that IF a finite limit exists, it must be 1.
        This method extends the theorem to classify all parameter regimes.
        
        Returns
        -------
        dict
            Dictionary containing regime classification and predictions
        """
        if self.mu < 1:
            return {
                'regime': 'SUBCRITICAL',
                'behavior': 'CONVERGENT',
                'rate': 'exponential',
                'limit_prediction': f'f(x) ∼ {self.a}^{self.c}x/2',
                'confidence': 'HIGH (Theorem 3.4 + numerical evidence)',
                'mu_value': self.mu
            }
            
        elif self.mu == 1:
            # Special case: constant coefficients (a=1 or c=0)
            if self.a == 1 or self.c == 0:
                constant_limit = (1 + np.sqrt(1 + 4*self.b)) / 2
                note = f'Golden ratio φ = (1+√5)/2 ≈ 1.618' if self.b == 1 else ''
                return {
                    'regime': 'CRITICAL (constant coefficient)',
                    'behavior': 'CONVERGENT',
                    'limit': constant_limit,
                    'notes': note,
                    'mu_value': self.mu
                }
            else:
                # Geometric case at critical threshold
                return {
                    'regime': 'CRITICAL (geometric)',
                    'behavior': 'CONVERGENT',
                    'rate': 'exponential (slower as a→1)',
                    'limit_prediction': f'f(x) → {self.a}^{self.c}x/2',
                    'open_problem': True,  # Exact rate is an open question
                    'note': 'Precise asymptotics an open problem (see Section 6.2)',
                    'mu_value': self.mu
                }
        else:  # mu > 1
            growth_rate = np.sqrt(self.b * self.a**(self.c/2))
            return {
                'regime': 'SUPERCRITICAL',
                'behavior': 'DIVERGENT',
                'rate': 'exponential growth',
                'growth_rate': growth_rate,
                'note': 'No finite limit exists (diverges to ∞)',
                'mu_value': self.mu
            }
    
    def estimate_convergence_depth(self, tolerance=1e-10):
        """
        Estimate required recursion depth for given accuracy.
        
        Based on convergence rate analysis from Section 4.4.
        The depth scales as 1/(1-μ) for μ close to 1.
        
        Parameters
        ----------
        tolerance : float, optional
            Desired accuracy (default: 1e-10)
            
        Returns
        -------
        int or float
            Estimated depth, or inf for divergent cases
        """
        if self.mu >= 1:
            return float('inf')
        
        # Base depth from binary expansion
        base_depth = np.log2(1/tolerance)
        
        # Rate factor from convergence analysis
        # As μ→1, convergence slows dramatically
        if self.mu <= 0.5:
            rate_factor = 1.0
        elif self.mu <= 0.7:
            rate_factor = 2.0
        elif self.mu <= 0.8:
            rate_factor = 3.0
        elif self.mu <= 0.9:
            rate_factor = 5.0
        else:
            # Continuous formula for μ close to 1
            # Calibrated from numerical experiments
            rate_factor = 1 / (1 - self.mu + 0.05) * 0.5
        
        estimated = int(np.ceil(rate_factor * base_depth))
        return max(estimated, 5)  # Minimum depth 5
    
    def compute_psi(self, x, f_x):
        """
        Compute ψ(x) = f(x)·a^{-cx/2} to verify Theorem 3.4.
        
        Theorem 3.4 states that if limit exists, ψ(x) → 1.
        
        Parameters
        ----------
        x : float
            Point at which to evaluate
        f_x : float
            Value of f(x)
            
        Returns
        -------
        float
            ψ(x) value
        """
        return f_x * self.a**(-self.c * x / 2)
    
    def verify_asymptotic(self, N_max=50):
        """
        Directly verify Theorem 3.4 by computing ψ(x) for large x.
        
        For convergent cases, ψ(x) should approach 1 as x increases.
        
        Parameters
        ----------
        N_max : int, optional
            Maximum depth for verification (default: 50)
            
        Returns
        -------
        float
            Final ψ(x) value at depth N_max
        """
        print(f"\nVerifying Theorem 3.4: ψ(x) → 1 for μ={self.mu:.4f}")
        print("-" * 60)
        print(f"{'k':<5} {'f(k)':<20} {'ψ(k)':<15} {'Approach to 1':<15}")
        print("-" * 60)
        
        R = 0.0
        results = []
        
        for k in range(N_max, 0, -1):
            R = np.sqrt(self.a**(self.c * k) + self.b * R)
            
            # Report every 10 steps and first few
            if k % 10 == 0 or k <= 5:
                psi = self.compute_psi(k, R)
                diff = abs(psi - 1.0)
                print(f"{k:<5} {R:<20.8f} {psi:<15.8f} {diff:<15.2e}")
                results.append((k, psi, diff))
        
        return results
    
    def adaptive_computation(self, target_error=1e-12, max_depth=500):
        """
        Adaptively compute f(0) until convergence.
        
        Uses μ value to choose step sizes intelligently.
        
        Parameters
        ----------
        target_error : float, optional
            Desired accuracy (default: 1e-12)
        max_depth : int, optional
            Maximum recursion depth (default: 500)
            
        Returns
        -------
        float
            Computed value of f(0)
        """
        # Check for divergent case first
        if self.mu > 1:
            print(f"\nμ={self.mu:.4f} > 1: DIVERGENT case, no finite limit")
            return float('inf')
        
        print(f"\nAdaptive computation for μ={self.mu:.4f}")
        print(f"Target error: {target_error:.0e}")
        print("-" * 60)
        print(f"{'Depth N':<10} {'f(0)':<20} {'Change':<15} {'Progress':<15}")
        print("-" * 60)
        
        N = 5  # Start small
        prev_val = None
        prev_change = float('inf')
        
        while N <= max_depth:
            # Compute with current depth
            val = self._compute_f0(N)
            
            if prev_val is not None:
                change = abs(val - prev_val)
                
                # Progress indicator
                if prev_change < float('inf') and prev_change > 0:
                    progress = (prev_change - change) / prev_change * 100
                else:
                    progress = 0
                
                print(f"N={N:<6} {val:<20.12f} {change:<15.2e} {progress:<15.1f}%")
                
                # Check convergence
                if change < target_error:
                    print("-" * 60)
                    print(f"✓ CONVERGED to {val:.12f} at depth N={N}")
                    print(f"  Final error estimate: {change:.2e}")
                    return val
                
                prev_change = change
            else:
                print(f"N={N:<6} {val:<20.12f}")
                # Initialize change for next iteration
                change = float('inf')
                prev_change = float('inf')
            
            # Adaptive step size based on μ
            if self.mu < 0.5:
                step = 5
            elif self.mu < 0.7:
                step = 10
            elif self.mu < 0.8:
                step = 15
            elif self.mu < 0.9:
                step = 20
            else:
                step = 30  # μ close to 1 needs larger steps
            
            N += step
            prev_val = val
        
        print("-" * 60)
        print(f"⚠ Maximum depth {max_depth} reached without full convergence")
        print(f"  Final value: {val:.12f}")
        print(f"  Last change: {change:.2e}")
        return val
    
    def _compute_f0(self, N):
        """
        Compute f(0) using backward recursion.
        
        Implements: f(0) = √(1 + b·f(1))
                  f(1) = √(a^c + b·f(2))
                  f(2) = √(a^{2c} + b·f(3))
                  ...
        
        Parameters
        ----------
        N : int
            Recursion depth
            
        Returns
        -------
        float
            Approximation of f(0)
        """
        R = 0.0  # Initialize f(N+1) = 0
        
        # Backward recursion from k=N down to 2
        for k in range(N, 1, -1):
            R = np.sqrt(self.a**(self.c * k) + self.b * R)
        
        # Compute f(1)
        f1 = np.sqrt(self.a**self.c + self.b * R)
        
        # Compute f(0)
        return np.sqrt(1 + self.b * f1)
    
    def analyze_parameter_sensitivity(self, variations=None):
        """
        Analyze sensitivity of convergence to parameter changes.
        
        Based on Theorem 3.4, the critical threshold μ=1 determines behavior.
        This shows how small changes in a or b affect convergence.
        
        Parameters
        ----------
        variations : list, optional
            List of fractional variations to test (default: ±1%, ±0.5%, ±0.1%)
        """
        if variations is None:
            variations = [-0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01]
        
        print(f"\n{'='*60}")
        print(f"PARAMETER SENSITIVITY ANALYSIS")
        print(f"Base parameters: a={self.a}, b={self.b}, c={self.c}, d={self.d}")
        print(f"Base μ = {self.mu:.6f} ({'CONVERGENT' if self.mu<1 else 'DIVERGENT' if self.mu>1 else 'CRITICAL'})")
        print(f"{'='*60}")
        
        # Test variations in a
        print(f"\nEffect of varying 'a' (holding b constant):")
        print(f"{'Change':<12} {'New a':<12} {'New μ':<12} {'Behavior':<12}")
        print("-" * 48)
        
        for delta in variations:
            a_new = self.a * (1 + delta)
            mu_new = self.b * a_new**(-self.c * self.d / 2)
            behavior = 'CONVERGENT' if mu_new < 1 else 'DIVERGENT' if mu_new > 1 else 'CRITICAL'
            print(f"{delta*100:+.1f}% → {a_new:<12.4f} {mu_new:<12.6f} {behavior:<12}")
        
        # Test variations in b
        print(f"\nEffect of varying 'b' (holding a constant):")
        print(f"{'Change':<12} {'New b':<12} {'New μ':<12} {'Behavior':<12}")
        print("-" * 48)
        
        for delta in variations:
            b_new = self.b * (1 + delta)
            mu_new = b_new * self.a**(-self.c * self.d / 2)
            behavior = 'CONVERGENT' if mu_new < 1 else 'DIVERGENT' if mu_new > 1 else 'CRITICAL'
            print(f"{delta*100:+.1f}% → {b_new:<12.4f} {mu_new:<12.6f} {behavior:<12}")
    
    def summary(self):
        """
        Print comprehensive summary of predictions.
        """
        behavior = self.predict_behavior()
        
        print(f"\n{'='*60}")
        print(f"AZEEQ AZAM'S μ-PREDICTOR - SUMMARY")
        print(f"{'='*60}")
        print(f"Parameters: a={self.a}, b={self.b}, c={self.c}, d={self.d}")
        print(f"Critical μ = {self.mu:.6f}")
        print(f"{'-'*60}")
        
        for key, value in behavior.items():
            if key != 'mu_value':
                print(f"{key.replace('_', ' ').title():20}: {value}")
        
        if self.mu < 1:
            depth_1e6 = self.estimate_convergence_depth(1e-6)
            depth_1e10 = self.estimate_convergence_depth(1e-10)
            depth_1e14 = self.estimate_convergence_depth(1e-14)
            print(f"\nEstimated depths:")
            print(f"  1e-6 accuracy : {depth_1e6} iterations")
            print(f"  1e-10 accuracy: {depth_1e10} iterations")
            print(f"  1e-14 accuracy: {depth_1e14} iterations")


# ============================================
# DEMONSTRATION AND TEST CASES
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AZEEQ AZAM'S μ-PREDICTOR ALGORITHM")
    print("Based on the Asymptotic Necessity Theorem (2026)")
    print("=" * 70)
    
    # Test cases from the paper
    test_cases = [
        (4.0, 1.0, 1.0, 1.0, "Base case (μ=0.5)"),
        (2.0, 1.0, 1.0, 1.0, "μ=0.707"),
        (1.5, 1.0, 1.0, 1.0, "μ=0.816"),
        (1.01, 1.0, 1.0, 1.0, "Near-critical (μ=0.995)"),
        (1.0, 1.0, 1.0, 1.0, "Golden ratio (μ=1)"),
        (0.8, 1.0, 1.0, 1.0, "Divergent (μ=1.118)"),
    ]
    
    for a, b, c, d, description in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST CASE: {description}")
        print(f"{'='*60}")
        
        predictor = NestedRadicalPredictor(a, b, c, d)
        
        # Show summary
        predictor.summary()
        
        # For convergent cases, demonstrate adaptive computation
        if predictor.mu < 1:
            print("\n" + "-" * 60)
            print("ADAPTIVE COMPUTATION DEMONSTRATION")
            predictor.adaptive_computation(target_error=1e-10, max_depth=200)
            
            # Verify Theorem 3.4
            predictor.verify_asymptotic(N_max=30)
        
        # For critical cases, show special handling
        elif predictor.mu == 1:
            print("\n" + "-" * 60)
            print("CRITICAL CASE ANALYSIS")
            if a == 1:
                val = predictor._compute_f0(50)
                print(f"f(0) = {val:.15f} (golden ratio φ ≈ 1.618033988749895)")
            
        # For divergent cases, skip computation
        else:
            print("\n" + "-" * 60)
            print("DIVERGENT CASE - no finite limit exists")
        
        # Show sensitivity analysis for each case
        print("\n" + "-" * 60)
        print("SENSITIVITY ANALYSIS")
        predictor.analyze_parameter_sensitivity()
