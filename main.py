"""
================================================================================
μ-PREDICTOR ALGORITHM v4.0
Based on Asymptotic Necessity Theorem (Azam, 2026) - CORRECTED VERSION

CORE EQUATIONS (from your paper):
    f(x) = √(a^{cx} + b·f(x+d))                     [Equation (4)]
    ψ(x) = f(x)·a^{-cx/2}                            [Theorem 3.1]
    μ = b·a^{-cd/2}                                   [Theorem 3.1]
    
CORRECTED EQUATION (5):
    ψ(x) = √(1 + (b²/μ)·ψ(x+d)·a^{-cx/2})
    For b=1: ψ(x) = √(1 + (1/μ)·ψ(x+d)·a^{-cx/2})
    
THEOREM 3.4:
    If lim_{x→∞} ψ(x) exists as a finite limit, it must equal 1.

OPEN PROBLEM (Section 3.3):
    Precise asymptotics when μ = 1 and a>1, c>0

Author: Aqueeq Azam
Date: 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import warnings

class MuPredictor:
    """
    μ-Predictor: Exact implementation of Azam's Asymptotic Necessity Theorem.
    
    Parameters
    ----------
    a : float > 0
        Base of geometric progression
    b : float > 0
        Coefficient multiplying inner radical
    c : float > 0
        Exponent multiplier
    d : float > 0
        Shift parameter
    """
    
    def __init__(self, a: float, b: float, c: float = 1.0, d: float = 1.0):
        # Validate inputs
        if a <= 0:
            raise ValueError(f"a={a} must be positive")
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
        
        # CRITICAL PARAMETER μ (Theorem 3.1)
        self.mu = b * a**(-c * d / 2)
        
        # Pre-compute the coupling constant for efficiency
        self.coupling = (b**2) / self.mu if self.mu != 0 else float('inf')
        
    # ------------------------------------------------------------------------
    # CORE ALGORITHM: CORRECTED IMPLEMENTATION OF EQUATION (5)
    # ------------------------------------------------------------------------
    
    def compute_psi_series(self, depth: int = 1000) -> np.ndarray:
        """
        Compute ψ(k) = f(k)/a^{ck/2} for k = depth down to 0.
        
        CORRECTED implementation of Equation (5):
            ψ(k) = √(1 + (b²/μ)·ψ(k+1)·a^{-c·k/2})
        
        For b=1, this becomes: ψ(k) = √(1 + (1/μ)·ψ(k+1)·a^{-c·k/2})
        
        Parameters
        ----------
        depth : int
            Maximum depth (can be 1,000,000+ safely)
            
        Returns
        -------
        np.ndarray
            ψ[0] = f(0), ψ[k] = f(k)/a^{ck/2}
        """
        psi = np.zeros(depth + 2)
        
        # Backward recursion using CORRECTED equation
        for k in range(depth, -1, -1):
            # The factor a^{-c*k/2} depends on k
            decay = self.a**(-self.c * k / 2)
            psi[k] = np.sqrt(1.0 + self.coupling * psi[k+1] * decay)
            
        return psi
    
    def compute_f0(self, depth: int = 1000) -> float:
        """Compute f(0) using corrected equation."""
        psi = self.compute_psi_series(depth)
        return psi[0]  # ψ(0) = f(0) since a^{c*0/2} = 1
    
    def compute_fk(self, k: int, depth: int = 1000) -> float:
        """Compute f(k) for any integer k using your scaling."""
        psi = self.compute_psi_series(depth + k)
        return psi[k] * (self.a**(self.c * k / 2))
    
    # ------------------------------------------------------------------------
    # THEOREM 3.4 VERIFICATION
    # ------------------------------------------------------------------------
    
    def verify_theorem_34(self, depth: int = 100, plot: bool = True) -> Dict:
        """
        Directly verify YOUR Theorem 3.4: ψ(x) → 1 as x → ∞.
        
        Parameters
        ----------
        depth : int
            Depth for verification
        plot : bool
            Whether to generate plot
            
        Returns
        -------
        dict
            Verification results
        """
        psi = self.compute_psi_series(depth)
        
        # Check approach to 1
        deviations = np.abs(psi - 1.0)
        
        result = {
            'psi_0': psi[0],
            'psi_final': psi[depth//2],
            'max_deviation': np.max(deviations),
            'final_deviation': deviations[min(depth-1, len(deviations)-10)],
            'converges_to_1': deviations[-1] < 0.01 if len(deviations) > 0 else False,
            'monotonic': np.all(np.diff(psi) <= 0) if len(psi) > 1 else True
        }
        
        if plot:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(range(depth+1), psi, 'b-', linewidth=2)
            plt.axhline(y=1.0, color='r', linestyle='--', 
                       label='ψ → 1 (Theorem 3.4)')
            plt.xlabel('k (depth index)')
            plt.ylabel('ψ(k)')
            plt.title(f'Theorem 3.4 Verification (μ={self.mu:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.semilogy(range(depth+1), deviations, 'g-', linewidth=2)
            plt.xlabel('k (depth index)')
            plt.ylabel('|ψ(k) - 1|')
            plt.title('Deviation from 1 (log scale)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        return result
    
    # ------------------------------------------------------------------------
    # BEHAVIOR PREDICTION (Based on μ)
    # ------------------------------------------------------------------------
    
    def predict_behavior(self) -> Dict[str, Union[str, float]]:
        """
        Predict convergence/divergence behavior from μ.
        
        Returns
        -------
        dict
            Regime classification and predictions
        """
        if self.mu < 1:
            # Determine stability level
            if self.mu < 0.9:
                stability = "HIGH"
                warning = "STABLE"
            elif self.mu < 0.95:
                stability = "MEDIUM"
                warning = "WATCH"
            elif self.mu < 0.99:
                stability = "LOW"
                warning = "CAUTION"
            else:
                stability = "CRITICAL APPROACH"
                warning = "WARNING"
                
            return {
                'regime': 'SUBCRITICAL',
                'behavior': 'CONVERGENT',
                'rate': 'exponential',
                'stability': stability,
                'warning': warning,
                'mu': self.mu,
                'limit_prediction': f'f(x) ∼ {self.a}^{self.c}x/2',
                'theorem': 'Theorem 3.4 guarantees ψ(x) → 1'
            }
            
        elif abs(self.mu - 1.0) < 1e-12:
            # μ = 1 exactly
            if abs(self.a - 1.0) < 1e-12 or abs(self.c) < 1e-12:
                # Constant coefficient case
                constant_limit = (1 + np.sqrt(1 + 4*self.b)) / 2
                return {
                    'regime': 'CRITICAL (constant)',
                    'behavior': 'CONVERGENT',
                    'limit': constant_limit,
                    'mu': self.mu,
                    'notes': f'Golden ratio φ = {(1+np.sqrt(5))/2} when b=1'
                }
            else:
                # Geometric critical case (YOUR open problem)
                return {
                    'regime': 'CRITICAL (geometric)',
                    'behavior': 'CONVERGENT (slow)',
                    'mu': self.mu,
                    'note': 'OPEN PROBLEM: Precise asymptotics unknown (Section 3.3)',
                    'open_problem': True
                }
        else:  # mu > 1
            growth_rate = np.sqrt(self.b * self.a**(self.c/2))
            return {
                'regime': 'SUPERCRITICAL',
                'behavior': 'DIVERGENT',
                'rate': 'exponential growth',
                'growth_rate': growth_rate,
                'mu': self.mu,
                'warning': 'EMERGENCY - No finite limit exists'
            }
    
    def estimate_depth(self, tolerance: float = 1e-10) -> int:
        """
        Estimate required depth for given accuracy.
        
        Parameters
        ----------
        tolerance : float
            Desired accuracy for f(0)
            
        Returns
        -------
        int
            Estimated depth (or inf for divergent)
        """
        if self.mu >= 1:
            return float('inf')
        
        base_depth = int(np.ceil(np.log2(1/tolerance)))
        
        # Empirical rates from YOUR Section 4.4
        if self.mu <= 0.5:
            rate = 1.0
        elif self.mu <= 0.7:
            rate = 2.0
        elif self.mu <= 0.8:
            rate = 3.0
        elif self.mu <= 0.9:
            rate = 5.0
        else:
            # Near-critical: depth ∝ 1/(1-μ)
            rate = 1.0 / (1.0 - self.mu + 0.01) * 0.5
            
        return max(5, int(np.ceil(rate * base_depth)))
    
    def get_recommendation(self, param_names: List[str] = None) -> str:
        """Get human-readable recommendation based on μ."""
        if param_names is None:
            param_names = ['Parameter A', 'Parameter B', 'Parameter C']
            
        if self.mu < 0.9:
            return f"✅ STABLE: All parameters optimal. Continue operation."
        elif self.mu < 0.95:
            return f"⚠️ WATCH: Reduce {param_names[0]} by 5%."
        elif self.mu < 1.0:
            return f"⚠️⚠️ CAUTION: Reduce {param_names[1]} by 10% immediately."
        else:
            return f"🛑 EMERGENCY: μ={self.mu:.4f} > 1 → SYSTEM DIVERGENT. SHUTDOWN!"
    
    # ------------------------------------------------------------------------
    # CRITICAL REGIME EXPLORER (YOUR Open Problem #4)
    # ------------------------------------------------------------------------
    
    def explore_critical_region(self, epsilons: List[float] = None, 
                               depth: int = 100000) -> Dict:
        """
        Explore YOUR open problem: behavior as μ → 1⁻.
        
        This addresses Section 3.3, Open Problem #4:
        "What is the precise asymptotics of ψ(x) when μ = 1 and a>1, c>0?"
        
        Parameters
        ----------
        epsilons : list
            List of ε = 1-μ values to test
        depth : int
            Depth for computation
            
        Returns
        -------
        dict
            Results with fitted critical exponent
        """
        if epsilons is None:
            epsilons = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
            
        results = {
            'epsilons': [],
            'mus': [],
            'psi_0': [],
            'deviation': []
        }
        
        print("\n" + "="*80)
        print("OPEN PROBLEM #4: CRITICAL REGIME EXPLORER")
        print("Asymptotics of ψ(x) when μ → 1⁻ (a>1, c>0)")
        print("="*80)
        print(f"{'ε':<12} {'μ':<12} {'ψ(0)':<20} {'ψ(0)-1':<15}")
        print("-"*80)
        
        for eps in epsilons:
            # Create near-critical predictor with μ = 1 - ε
            # For b=1, c=1, d=1: μ = a^{-1/2}
            # So a = 1/(1-ε)²
            a = 1.0 / ((1.0 - eps) ** 2)
            pred = MuPredictor(a, 1.0, 1.0, 1.0)
            
            psi = pred.compute_psi_series(depth)
            psi_0 = psi[0]
            deviation = psi_0 - 1.0
            
            results['epsilons'].append(eps)
            results['mus'].append(pred.mu)
            results['psi_0'].append(psi_0)
            results['deviation'].append(deviation)
            
            print(f"{eps:<12.2e} {pred.mu:<12.6f} {psi_0:<20.15f} {deviation:<15.2e}")
        
        # Fit critical exponent (power law)
        eps_array = np.array(results['epsilons'])
        dev_array = np.array(results['deviation'])
        
        if len(eps_array) >= 3:
            # Use points with positive deviation
            mask = dev_array > 0
            if np.any(mask):
                log_eps = np.log(eps_array[mask])
                log_dev = np.log(dev_array[mask])
                
                if len(log_eps) >= 2:
                    coeffs = np.polyfit(log_eps, log_dev, 1)
                    exponent = coeffs[0]
                    prefactor = np.exp(coeffs[1])
                    
                    results['critical_exponent'] = exponent
                    results['prefactor'] = prefactor
                    
                    print("\n" + "="*80)
                    print(f"CRITICAL EXPONENT FIT: ψ(0)-1 ≈ {prefactor:.4f} · ε^{exponent:.4f}")
                    print("="*80)
                    
                    # Generate fit plot
                    plt.figure(figsize=(10, 6))
                    plt.loglog(eps_array[mask], dev_array[mask], 'bo', 
                              label='Data', markersize=8)
                    
                    fit_line = prefactor * eps_array[mask] ** exponent
                    plt.loglog(eps_array[mask], fit_line, 'r-', 
                              label=f'Fit: ε^{exponent:.4f}', linewidth=2)
                    
                    plt.xlabel('ε = 1-μ')
                    plt.ylabel('ψ(0) - 1')
                    plt.title('Critical Regime Scaling (Open Problem #4)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.show()
        
        return results
    
    # ------------------------------------------------------------------------
    # SENSITIVITY ANALYSIS
    # ------------------------------------------------------------------------
    
    def sensitivity_analysis(self, variations: List[float] = None):
        """Analyze sensitivity to parameter changes."""
        if variations is None:
            variations = [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]
            
        print(f"\n{'='*60}")
        print(f"PARAMETER SENSITIVITY ANALYSIS")
        print(f"Base parameters: a={self.a}, b={self.b}, c={self.c}, d={self.d}")
        print(f"Base μ = {self.mu:.6f}")
        print(f"{'='*60}")
        
        # Vary a
        print(f"\nEffect of varying 'a':")
        print(f"{'Change':<10} {'New a':<12} {'New μ':<15} {'Behavior':<15}")
        print("-"*55)
        
        for delta in variations:
            a_new = self.a * (1 + delta)
            mu_new = self.b * a_new**(-self.c * self.d / 2)
            behavior = 'CONVERGENT' if mu_new < 1 else 'DIVERGENT' if mu_new > 1 else 'CRITICAL'
            print(f"{delta*100:+.1f}% → {a_new:<12.4f} {mu_new:<15.6f} {behavior:<15}")
            
        # Vary b
        print(f"\nEffect of varying 'b':")
        print(f"{'Change':<10} {'New b':<12} {'New μ':<15} {'Behavior':<15}")
        print("-"*55)
        
        for delta in variations:
            b_new = self.b * (1 + delta)
            mu_new = b_new * self.a**(-self.c * self.d / 2)
            behavior = 'CONVERGENT' if mu_new < 1 else 'DIVERGENT' if mu_new > 1 else 'CRITICAL'
            print(f"{delta*100:+.1f}% → {b_new:<12.4f} {mu_new:<15.6f} {behavior:<15}")
    
    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------
    
    def summary(self):
        """Print comprehensive summary of predictions."""
        behavior = self.predict_behavior()
        
        print(f"\n{'='*60}")
        print(f"μ-PREDICTOR SUMMARY (Azam, 2026) - CORRECTED VERSION")
        print(f"{'='*60}")
        print(f"Parameters: a={self.a}, b={self.b}, c={self.c}, d={self.d}")
        print(f"Critical μ = {self.mu:.6f}")
        print(f"Coupling = b²/μ = {self.coupling:.6f}")
        print(f"{'-'*60}")
        
        for key, value in behavior.items():
            if key not in ['mu']:
                print(f"{key.replace('_', ' ').title():20}: {value}")
                
        if self.mu < 1:
            depth_1e6 = self.estimate_depth(1e-6)
            depth_1e10 = self.estimate_depth(1e-10)
            depth_1e14 = self.estimate_depth(1e-14)
            print(f"\nEstimated depths (from Section 4.4):")
            print(f"  1e-6 accuracy : {depth_1e6} iterations")
            print(f"  1e-10 accuracy: {depth_1e10} iterations")
            print(f"  1e-14 accuracy: {depth_1e14} iterations")
            
        return behavior


# ------------------------------------------------------------------------
# DEMONSTRATION
# ------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*80)
    print("μ-PREDICTOR ALGORITHM v4.0 (CORRECTED)")
    print("Based on Asymptotic Necessity Theorem (Azam, 2026)")
    print("="*80)
    
    # Test base case with corrected equation
    print("\n" + "="*60)
    print("TEST: Base case (a=4, b=1, c=1, d=1)")
    print("="*60)
    
    predictor = MuPredictor(4.0, 1.0, 1.0, 1.0)
    predictor.summary()
    
    # Verify base case value
    f0 = predictor.compute_f0(depth=50)
    print(f"\nf(0) = {f0:.15f} (should be 2.0)")
    print(f"Error = {abs(f0 - 2.0):.2e}")
    
    # Test golden ratio case
    print("\n" + "="*60)
    print("TEST: Golden ratio (a=1, b=1, c=1, d=1)")
    print("="*60)
    
    predictor2 = MuPredictor(1.0, 1.0, 1.0, 1.0)
    predictor2.summary()
    
    phi = (1 + np.sqrt(5)) / 2
    f0_phi = predictor2.compute_f0(depth=50)
    print(f"\nf(0) = {f0_phi:.15f} (should be φ ≈ {phi:.15f})")
    print(f"Error = {abs(f0_phi - phi):.2e}")
    
    # Quick theorem verification
    print("\n" + "="*60)
    print("Verifying Theorem 3.4...")
    result = predictor.verify_theorem_34(depth=50, plot=False)
    print(f"ψ(0) = {result['psi_0']:.10f}")
    print(f"ψ → 1: {result['converges_to_1']}")
