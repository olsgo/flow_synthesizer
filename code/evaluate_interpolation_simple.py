#!/usr/bin/env python3
"""
Parameter Space Interpolation Testing Tool for RegressionAE

This tool evaluates how well the model handles parameter space interpolation by:
1. Testing smooth transitions between input spectrograms
2. Analyzing parameter prediction smoothness during interpolation
3. Detecting discontinuities and artifacts in parameter predictions
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
import argparse
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

# Import model and data utilities
import sys
sys.path.append('.')
from models.vae.ae import RegressionAE
from utils.data import load_dataset

class InterpolationEvaluator:
    def __init__(self, model_path, dataset_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.dataset_path = dataset_path
        
        # Load parameter schema
        with open('params_schema.json', 'r') as f:
            self.param_schema = json.load(f)
        
        self.param_names = self.param_schema['parameter_order']
        self.num_params = len(self.param_names)
        
        print(f"Interpolation evaluation for {self.num_params} parameters")
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}")
        
        model = torch.load(self.model_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        
        return model
        
    def get_sample_pairs(self, dataloader, num_pairs=20):
        """Get pairs of samples for interpolation testing"""
        samples = []
        targets = []
        
        with torch.no_grad():
            for data, params, meta, audio in dataloader:
                samples.append(data)
                targets.append(params)
                if len(samples) * data.shape[0] >= num_pairs * 2:
                    break
                    
        # Concatenate and select pairs
        all_samples = torch.cat(samples, dim=0)[:num_pairs * 2]
        all_targets = torch.cat(targets, dim=0)[:num_pairs * 2]
        
        # Create pairs
        pairs = []
        for i in range(0, len(all_samples), 2):
            if i + 1 < len(all_samples):
                pairs.append({
                    'sample_a': all_samples[i],
                    'sample_b': all_samples[i + 1],
                    'target_a': all_targets[i],
                    'target_b': all_targets[i + 1]
                })
                
        return pairs
        
    def linear_interpolation(self, start, end, num_steps):
        """Perform linear interpolation between two points"""
        alphas = np.linspace(0, 1, num_steps)
        interpolated = []
        
        for alpha in alphas:
            interpolated_point = (1 - alpha) * start + alpha * end
            interpolated.append(interpolated_point)
            
        return torch.stack(interpolated)
        
    def evaluate_input_interpolation(self, model, sample_a, sample_b, target_a, target_b, num_steps=10):
        """Evaluate interpolation in input space"""
        with torch.no_grad():
            # Interpolate in input space
            interpolated_inputs = self.linear_interpolation(sample_a, sample_b, num_steps)
            
            # Get model predictions for interpolated inputs
            predicted_params = []
            for inp in interpolated_inputs:
                inp = inp.unsqueeze(0).to(self.device)
                pred = model(inp)
                predicted_params.append(pred.squeeze().cpu())
                
            predicted_params = torch.stack(predicted_params)
            
            # Ground truth parameter interpolation
            ground_truth_params = self.linear_interpolation(target_a, target_b, num_steps)
            
            return {
                'interpolated_inputs': interpolated_inputs,
                'predicted_params': predicted_params,
                'ground_truth_params': ground_truth_params,
                'target_a': target_a,
                'target_b': target_b
            }
            
    def calculate_smoothness_metrics(self, sequence):
        """Calculate smoothness metrics for a parameter sequence"""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.numpy()
            
        # First and second derivatives
        first_diff = np.diff(sequence, axis=0)
        second_diff = np.diff(first_diff, axis=0)
        
        # Total variation (sum of absolute first differences)
        total_variation = np.sum(np.abs(first_diff), axis=0)
        
        # Smoothness score (inverse of total variation)
        smoothness = 1.0 / (1.0 + np.mean(total_variation))
        
        # Acceleration (second derivative magnitude)
        acceleration = np.mean(np.abs(second_diff), axis=0)
        
        return {
            'total_variation': np.mean(total_variation),
            'smoothness_score': smoothness,
            'mean_acceleration': np.mean(acceleration),
            'per_param_variation': total_variation,
            'per_param_acceleration': acceleration
        }
        
    def detect_discontinuities(self, sequence, threshold_factor=3.0):
        """Detect discontinuities in parameter sequence"""
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.numpy()
            
        first_diff = np.diff(sequence, axis=0)
        
        # Calculate threshold based on standard deviation
        std_diff = np.std(first_diff, axis=0)
        threshold = threshold_factor * std_diff
        
        # Find discontinuities
        discontinuities = []
        for i, diff in enumerate(first_diff):
            for j, (d, t) in enumerate(zip(diff, threshold)):
                if abs(d) > t:
                    discontinuities.append({
                        'step': i,
                        'parameter': self.param_names[j],
                        'magnitude': abs(d),
                        'threshold': t
                    })
                    
        return discontinuities
        
    def evaluate_interpolation_quality(self, model, pairs, num_steps=10):
        """Evaluate interpolation quality for multiple pairs"""
        results = []
        
        print(f"Evaluating interpolation quality for {len(pairs)} pairs...")
        
        for i, pair in enumerate(pairs):
            print(f"Processing pair {i+1}/{len(pairs)}")
            
            # Input space interpolation
            interp_results = self.evaluate_input_interpolation(
                model, pair['sample_a'], pair['sample_b'], 
                pair['target_a'], pair['target_b'], num_steps
            )
            
            # Calculate reconstruction error
            reconstruction_error = mean_squared_error(
                interp_results['ground_truth_params'].numpy(),
                interp_results['predicted_params'].numpy()
            )
            
            # Calculate smoothness metrics
            pred_smoothness = self.calculate_smoothness_metrics(interp_results['predicted_params'])
            gt_smoothness = self.calculate_smoothness_metrics(interp_results['ground_truth_params'])
            
            # Detect discontinuities
            pred_discontinuities = self.detect_discontinuities(interp_results['predicted_params'])
            gt_discontinuities = self.detect_discontinuities(interp_results['ground_truth_params'])
            
            # Calculate parameter space distance
            param_distance = euclidean(
                pair['target_a'].numpy(),
                pair['target_b'].numpy()
            )
            
            pair_result = {
                'pair_idx': i,
                'reconstruction_error': reconstruction_error,
                'parameter_distance': param_distance,
                'predicted_smoothness': pred_smoothness,
                'ground_truth_smoothness': gt_smoothness,
                'predicted_discontinuities': len(pred_discontinuities),
                'ground_truth_discontinuities': len(gt_discontinuities),
                'smoothness_ratio': pred_smoothness['smoothness_score'] / gt_smoothness['smoothness_score'],
                'discontinuity_details': pred_discontinuities
            }
            
            results.append(pair_result)
            
        return results
        
    def create_interpolation_report(self, results, output_dir):
        """Create comprehensive interpolation quality report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("PARAMETER SPACE INTERPOLATION EVALUATION REPORT")
        print("="*80)
        
        # Overall statistics
        reconstruction_errors = [r['reconstruction_error'] for r in results]
        smoothness_ratios = [r['smoothness_ratio'] for r in results]
        pred_discontinuities = [r['predicted_discontinuities'] for r in results]
        gt_discontinuities = [r['ground_truth_discontinuities'] for r in results]
        
        print(f"\nINTERPOLATION QUALITY METRICS:")
        print(f"Reconstruction Error - Mean: {np.mean(reconstruction_errors):.6f}, Std: {np.std(reconstruction_errors):.6f}")
        print(f"Smoothness Ratio - Mean: {np.mean(smoothness_ratios):.4f}, Std: {np.std(smoothness_ratios):.4f}")
        print(f"Predicted Discontinuities - Mean: {np.mean(pred_discontinuities):.2f}, Total: {np.sum(pred_discontinuities)}")
        print(f"Ground Truth Discontinuities - Mean: {np.mean(gt_discontinuities):.2f}, Total: {np.sum(gt_discontinuities)}")
        
        # Parameter-specific analysis
        param_errors = {param: [] for param in self.param_names}
        for result in results:
            for detail in result['discontinuity_details']:
                param_errors[detail['parameter']].append(detail['magnitude'])
                
        print(f"\nPARAMETER-SPECIFIC DISCONTINUITIES:")
        for param, errors in param_errors.items():
            if errors:
                print(f"  {param}: {len(errors)} discontinuities, avg magnitude: {np.mean(errors):.4f}")
                
        # Create visualizations
        self.create_interpolation_visualizations(results, output_dir)
        
        # Save detailed results
        results_df = pd.DataFrame([
            {
                'pair_idx': r['pair_idx'],
                'reconstruction_error': r['reconstruction_error'],
                'parameter_distance': r['parameter_distance'],
                'smoothness_ratio': r['smoothness_ratio'],
                'predicted_discontinuities': r['predicted_discontinuities'],
                'ground_truth_discontinuities': r['ground_truth_discontinuities']
            }
            for r in results
        ])
        
        results_df.to_csv(output_dir / 'interpolation_results.csv', index=False)
        
        # Save full results as JSON
        with open(output_dir / 'interpolation_detailed_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = []
            for r in results:
                json_r = r.copy()
                json_r['predicted_smoothness'] = {k: float(v) if isinstance(v, np.ndarray) and v.size == 1 else (v.tolist() if isinstance(v, np.ndarray) else v) 
                                                for k, v in r['predicted_smoothness'].items()}
                json_r['ground_truth_smoothness'] = {k: float(v) if isinstance(v, np.ndarray) and v.size == 1 else (v.tolist() if isinstance(v, np.ndarray) else v) 
                                                   for k, v in r['ground_truth_smoothness'].items()}
                json_results.append(json_r)
            json.dump(json_results, f, indent=2, default=str)
            
        return results_df
        
    def create_interpolation_visualizations(self, results, output_dir):
        """Create visualizations for interpolation analysis"""
        plt.figure(figsize=(15, 10))
        
        # 1. Reconstruction error vs parameter distance
        plt.subplot(2, 3, 1)
        errors = [r['reconstruction_error'] for r in results]
        distances = [r['parameter_distance'] for r in results]
        
        plt.scatter(distances, errors, alpha=0.6)
        plt.xlabel('Parameter Space Distance')
        plt.ylabel('Reconstruction Error')
        plt.title('Error vs Parameter Distance')
        
        # 2. Smoothness ratio distribution
        plt.subplot(2, 3, 2)
        smoothness_ratios = [r['smoothness_ratio'] for r in results]
        plt.hist(smoothness_ratios, bins=15, alpha=0.7, edgecolor='black')
        plt.xlabel('Smoothness Ratio (Pred/GT)')
        plt.ylabel('Count')
        plt.title('Smoothness Ratio Distribution')
        plt.axvline(1.0, color='red', linestyle='--', label='Perfect Smoothness')
        plt.legend()
        
        # 3. Discontinuities comparison
        plt.subplot(2, 3, 3)
        pred_disc = [r['predicted_discontinuities'] for r in results]
        gt_disc = [r['ground_truth_discontinuities'] for r in results]
        
        x = range(len(results))
        plt.bar([i - 0.2 for i in x], pred_disc, width=0.4, label='Predicted', alpha=0.7)
        plt.bar([i + 0.2 for i in x], gt_disc, width=0.4, label='Ground Truth', alpha=0.7)
        plt.xlabel('Pair Index')
        plt.ylabel('Number of Discontinuities')
        plt.title('Discontinuities by Pair')
        plt.legend()
        
        # 4. Error distribution
        plt.subplot(2, 3, 4)
        plt.hist(errors, bins=15, alpha=0.7, edgecolor='black')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.title('Reconstruction Error Distribution')
        plt.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
        plt.legend()
        
        # 5. Smoothness vs Error
        plt.subplot(2, 3, 5)
        plt.scatter(smoothness_ratios, errors, alpha=0.6)
        plt.xlabel('Smoothness Ratio')
        plt.ylabel('Reconstruction Error')
        plt.title('Smoothness vs Error')
        
        # 6. Parameter-specific discontinuity count
        plt.subplot(2, 3, 6)
        param_disc_counts = {param: 0 for param in self.param_names}
        for result in results:
            for detail in result['discontinuity_details']:
                param_disc_counts[detail['parameter']] += 1
                
        # Show top 10 most problematic parameters
        sorted_params = sorted(param_disc_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        if sorted_params:
            params, counts = zip(*sorted_params)
            plt.bar(range(len(params)), counts)
            plt.xticks(range(len(params)), params, rotation=45, ha='right')
            plt.ylabel('Discontinuity Count')
            plt.title('Top 10 Problematic Parameters')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'interpolation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Interpolation visualizations saved to {output_dir}")
        
    def run_evaluation(self, num_pairs=20, num_steps=10, output_dir='evaluation_results/interpolation'):
        """Run the complete interpolation evaluation"""
        print("Starting Parameter Space Interpolation Evaluation...")
        
        # Load model
        model = self.load_model()
        
        # Get test dataloader
        args = argparse.Namespace(
            path=self.dataset_path,
            dataset='polymax_dataset',
            data='mel',
            batch_size=16,
            nbworkers=4,
            train_type='sequential'
        )
        
        train_loader, val_loader, test_loader, args = load_dataset(args)
        
        # Get sample pairs
        pairs = self.get_sample_pairs(test_loader, num_pairs)
        print(f"Generated {len(pairs)} sample pairs for interpolation testing")
        
        # Evaluate interpolation quality
        results = self.evaluate_interpolation_quality(model, pairs, num_steps)
        
        # Create report
        report_df = self.create_interpolation_report(results, output_dir)
        
        print(f"\nInterpolation evaluation complete! Results saved to {output_dir}")
        return results, report_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate parameter space interpolation quality')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, default='/Users/gjb/Datasets/polymax', help='Path to dataset')
    parser.add_argument('--num_pairs', type=int, default=20, help='Number of sample pairs to test')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/interpolation', help='Output directory')
    
    args = parser.parse_args()
    
    evaluator = InterpolationEvaluator(
        model_path=args.model_path,
        dataset_path=args.dataset_path
    )
    
    results, report_df = evaluator.run_evaluation(args.num_pairs, args.num_steps, args.output_dir)
    
if __name__ == '__main__':
    main()