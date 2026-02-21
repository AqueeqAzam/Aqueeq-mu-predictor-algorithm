# μ-Predictor: Algorithm for Infinite Geometric Nested Radicals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2402.xxxxx-red)](https://arxiv.org)

## 📖 Overview

This package implements the **μ-Predictor Algorithm** based on Azam's Asymptotic Necessity Theorem (2026). It predicts and computes infinite nested radicals of the form:

$$f(x) = \sqrt{a^{cx} + b \cdot f(x+d)}$$

The algorithm uses a single critical parameter $\mu = b \cdot a^{-cd/2}$ to predict convergence behavior **before computation**.

## 🚀 Key Innovations

| Feature | What It Does |
|---------|--------------|
| **μ-Based Prediction** | Single parameter μ predicts convergent/critical/divergent behavior |
| **Depth Estimation** | Predicts required iterations BEFORE computing |
| **Theorem Verification** | Directly verifies ψ(x) → 1 (Theorem 3.4) |
| **Sensitivity Analysis** | Shows how parameter changes affect convergence |
| **Critical Explorer** | Investigates open problem of μ → 1⁻ asymptotics |

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mu-predictor.git
cd mu-predictor

# Install dependencies
pip install -r requirements.txt
