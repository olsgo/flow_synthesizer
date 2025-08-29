#!/usr/bin/env python3
"""
Parameter Space Interpolation Testing Tool (Wrapper)

This script delegates to evaluate_interpolation_simple.InterpolationEvaluator,
which is compatible with RegressionAE-style models used in this project.
"""

import argparse
import sys
sys.path.append('.')
from evaluate_interpolation_simple import InterpolationEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate parameter space interpolation quality')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.model)')
    parser.add_argument('--dataset_path', type=str, default='datasets', help='Path to dataset')
    parser.add_argument('--num_pairs', type=int, default=20, help='Number of sample pairs to test')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/interpolation', help='Output directory')

    args = parser.parse_args()

    evaluator = InterpolationEvaluator(
        model_path=args.model_path,
        dataset_path=args.dataset_path
    )
    evaluator.run_evaluation(args.num_pairs, args.num_steps, args.output_dir)


if __name__ == '__main__':
    main()

