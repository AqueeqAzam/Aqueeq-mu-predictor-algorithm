from mu_predictor import MuPredictor
import numpy as np

pred = MuPredictor(1.0, 1.0, 1.0, 1.0)
phi = (1 + np.sqrt(5)) / 2

f0 = pred.compute_f0(50)
print(f"φ = {phi:.15f}")
print(f"f(0) = {f0:.15f}")
