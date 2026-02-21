from mu_predictor import MuPredictor

pred = MuPredictor(4.0, 1.0, 1.0, 1.0)
pred.summary()

f0 = pred.compute_f0(50)
print(f"\nf(0) = {f0:.15f} (should be 2.0)")
