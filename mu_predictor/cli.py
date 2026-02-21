"""Command-line interface for μ-Predictor."""

import argparse
from .core import MuPredictor

def main():
    parser = argparse.ArgumentParser(description='μ-Predictor CLI')
    parser.add_argument('--a', type=float, required=True, help='Parameter a')
    parser.add_argument('--b', type=float, required=True, help='Parameter b')
    parser.add_argument('--c', type=float, default=1.0, help='Parameter c')
    parser.add_argument('--d', type=float, default=1.0, help='Parameter d')
    parser.add_argument('--depth', type=int, default=50, help='Computation depth')
    
    args = parser.parse_args()
    
    pred = MuPredictor(args.a, args.b, args.c, args.d)
    pred.summary()
    
    f0 = pred.compute_f0(args.depth)
    print(f"\nf(0) = {f0:.15f}")

if __name__ == '__main__':
    main()
