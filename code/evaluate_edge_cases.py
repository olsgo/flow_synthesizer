#!/usr/bin/env python3
"""
Edge Case Detection and Testing Tool

This tool evaluates model performance on edge cases by:
1. Testing extreme parameter values at boundaries
2. Generating unusual parameter combinations
3. Testing adversarial parameter sets
4. Evaluating model robustness to outliers
5. Identifying failure modes and limitations
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import argparse
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Import model and data utilities
import sys
sys.path.append('.')
from models.vae.ae import RegressionAE
from utils.data import load_dataset

class EdgeCaseEvaluator:
    def __init__(self, model_path, dataset_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.dataset_path = dataset_path
        
        # Load parameter schema
        with open('params_schema.json', 'r') as f:
            self.param_schema = json.load(f)
        
        self.param_names = self.param_schema['parameter_order']
        self.num_params = len(self.param_names)
        
        print(f"Edge case evaluation for {self.num_params} parameters")
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}")
        
        model = torch.load(self.model_path, map_location=self.device)
        model.eval()
        
        return model
        
    def get_dataset_statistics(self):
        """Get statistics from the training dataset"""
        print("Analyzing dataset statistics...")
        
        # Create args for dataset loading
        import argparse
        args = argparse.Namespace(
            path='/Users/gjb/Projects/flow_synthesizer/code/datasets',
            dataset='polymax_dataset',
            data='mel',
            batch_size=64,
            nbworkers=4,
            train_type='train'
        )
        
        # Change to parent directory temporarily
        import os
        original_cwd = os.getcwd()
        os.chdir('/Users/gjb/Projects/flow_synthesizer')
        try:
            train_loader, _, _, _ = load_dataset(args)
        finally:
            os.chdir(original_cwd)
        
        all_targets = []
        for data, params, meta, audio in train_loader:
            all_targets.append(params)
            
        dataset_params = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate statistics for each parameter
        param_stats = {}
        for i, param_name in enumerate(self.param_names):
            values = dataset_params[:, i]
            param_stats[param_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q01': np.percentile(values, 1),
                'q99': np.percentile(values, 99),
                'median': np.median(values)
            }
            
        return param_stats, dataset_params
        
    def generate_boundary_cases(self, param_stats):
        """Generate test cases at parameter boundaries"""
        print("Generating boundary test cases...")
        
        boundary_cases = []
        
        for param_name in self.param_names:
            expected_range = [0, 1]  # Default range for normalized parameters
            stats = param_stats[param_name]
            
            # Create base parameter set (median values)
            base_params = np.array([param_stats[p]['median'] for p in self.param_names])
            param_idx = self.param_names.index(param_name)
            
            # Test cases for this parameter
            test_values = [
                expected_range[0],  # Minimum boundary
                expected_range[1],  # Maximum boundary
                stats['q01'],       # 1st percentile
                stats['q99'],       # 99th percentile
                stats['min'],       # Dataset minimum
                stats['max']        # Dataset maximum
            ]
            
            for test_value in test_values:
                test_params = base_params.copy()
                test_params[param_idx] = test_value
                
                boundary_cases.append({
                    'type': 'boundary',
                    'param_name': param_name,
                    'test_value': test_value,
                    'parameters': test_params,
                    'description': f'{param_name} = {test_value:.4f}'
                })
                
        return boundary_cases
        
    def generate_extreme_combinations(self, param_stats, num_combinations=50):
        """Generate extreme parameter combinations"""
        print(f"Generating {num_combinations} extreme combinations...")
        
        extreme_cases = []
        
        for i in range(num_combinations):
            # Create random extreme combination
            extreme_params = np.zeros(self.num_params)
            
            for j, param_name in enumerate(self.param_names):
                expected_range = self.param_schema.get(param_name, {}).get('range', [0, 1])
                stats = param_stats[param_name]
                
                # Randomly choose extreme value (boundary or outlier)
                choice = np.random.choice(['min_boundary', 'max_boundary', 'outlier_low', 'outlier_high'])
                
                if choice == 'min_boundary':
                    value = expected_range[0]
                elif choice == 'max_boundary':
                    value = expected_range[1]
                elif choice == 'outlier_low':
                    # Value below 1st percentile
                    value = stats['q01'] - (stats['q01'] - stats['min']) * 0.5
                else:  # outlier_high
                    # Value above 99th percentile
                    value = stats['q99'] + (stats['max'] - stats['q99']) * 0.5
                    
                # Clamp to valid range
                value = np.clip(value, expected_range[0], expected_range[1])
                extreme_params[j] = value
                
            extreme_cases.append({
                'type': 'extreme_combination',
                'parameters': extreme_params,
                'description': f'Extreme combination #{i+1}'
            })
            
        return extreme_cases
        
    def generate_adversarial_cases(self, param_stats, dataset_params, num_cases=30):
        """Generate adversarial test cases"""
        print(f"Generating {num_cases} adversarial cases...")
        
        adversarial_cases = []
        
        # 1. Anti-correlated combinations
        # Find parameters that are typically correlated and create anti-correlated versions
        correlation_matrix = np.corrcoef(dataset_params.T)
        
        strong_correlations = []
        for i in range(len(self.param_names)):
            for j in range(i+1, len(self.param_names)):
                if abs(correlation_matrix[i, j]) > 0.5:
                    strong_correlations.append((i, j, correlation_matrix[i, j]))
                    
        # Create anti-correlated cases
        for i, (param1_idx, param2_idx, corr) in enumerate(strong_correlations[:10]):
            base_params = np.array([param_stats[p]['median'] for p in self.param_names])
            
            param1_name = self.param_names[param1_idx]
            param2_name = self.param_names[param2_idx]
            
            # If positively correlated, make one high and one low
            if corr > 0:
                base_params[param1_idx] = param_stats[param1_name]['q99']
                base_params[param2_idx] = param_stats[param2_name]['q01']
            else:
                # If negatively correlated, make both high or both low
                base_params[param1_idx] = param_stats[param1_name]['q99']
                base_params[param2_idx] = param_stats[param2_name]['q99']
                
            adversarial_cases.append({
                'type': 'anti_correlated',
                'parameters': base_params,
                'description': f'Anti-correlated: {param1_name} vs {param2_name}'
            })
            
        # 2. Outlier combinations
        # Parameters that are rarely seen together in extreme values
        for i in range(num_cases - len(adversarial_cases)):
            outlier_params = np.array([param_stats[p]['median'] for p in self.param_names])
            
            # Randomly select 2-3 parameters to set to extreme values
            num_extreme = np.random.choice([2, 3])
            extreme_indices = np.random.choice(self.num_params, num_extreme, replace=False)
            
            for idx in extreme_indices:
                param_name = self.param_names[idx]
                # Randomly choose extreme (high or low)
                if np.random.random() > 0.5:
                    outlier_params[idx] = param_stats[param_name]['q99']
                else:
                    outlier_params[idx] = param_stats[param_name]['q01']
                    
            adversarial_cases.append({
                'type': 'outlier_combination',
                'parameters': outlier_params,
                'description': f'Outlier combination #{i+1}'
            })
            
        return adversarial_cases
        
    def generate_synthetic_audio_features(self, parameters):
        """Generate synthetic mel spectrogram features for given parameters"""
        # This is a simplified approach - in practice, you'd want to use the actual
        # PolyMAX synthesizer to generate audio and extract mel spectrograms
        
        # For now, create synthetic features based on parameter relationships
        # This allows testing the model's parameter prediction capabilities
        
        # Use a simple linear combination with some nonlinearity
        np.random.seed(42)  # For reproducibility
        
        # Create synthetic mel spectrogram with proper dimensions (128, 173)
        mel_height, mel_width = 128, 173
        feature_dim = mel_height * mel_width
        
        # Random projection matrix (this would be learned from real data)
        projection_matrix = np.random.randn(self.num_params, feature_dim) * 0.1
        
        # Add some nonlinear transformations
        features = np.dot(parameters.reshape(1, -1), projection_matrix)
        features = features + 0.1 * np.sin(features * 2 * np.pi)
        features = features + 0.05 * np.random.randn(*features.shape)
        
        # Ensure positive values (like mel spectrograms)
        features = np.abs(features)
        
        # Reshape to mel spectrogram format and add batch dimension
        features = features.reshape(1, 1, mel_height, mel_width)  # (batch, channels, height, width)
        
        return torch.FloatTensor(features)
        
    def evaluate_edge_case(self, model, test_case):
        """Evaluate model performance on a single edge case"""
        parameters = test_case['parameters']
        
        # Generate synthetic audio features
        synthetic_features = self.generate_synthetic_audio_features(parameters)
        
        with torch.no_grad():
            synthetic_features = synthetic_features.to(self.device)
            
            # Get model prediction (RegressionAE returns only parameter predictions)
            predicted_params = model(synthetic_features)
            predicted_params = predicted_params.cpu().numpy().flatten()
            
        # Calculate errors
        mae = mean_absolute_error(parameters, predicted_params)
        mse = mean_squared_error(parameters, predicted_params)
        
        # Per-parameter errors
        param_errors = np.abs(parameters - predicted_params)
        
        # Identify problematic parameters
        error_threshold = np.percentile(param_errors, 90)  # Top 10% errors
        problematic_params = [
            {
                'name': self.param_names[i],
                'true_value': parameters[i],
                'predicted_value': predicted_params[i],
                'error': param_errors[i]
            }
            for i in range(len(param_errors))
            if param_errors[i] > error_threshold
        ]
        
        return {
            'mae': mae,
            'mse': mse,
            'max_error': np.max(param_errors),
            'true_parameters': parameters.tolist(),
            'predicted_parameters': predicted_params.tolist(),
            'param_errors': param_errors.tolist(),
            'problematic_params': problematic_params
        }
        
    def run_edge_case_evaluation(self, model, test_cases):
        """Run evaluation on all edge cases"""
        print(f"Evaluating {len(test_cases)} edge cases...")
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            if i % 20 == 0:
                print(f"Processing case {i+1}/{len(test_cases)}")
                
            evaluation = self.evaluate_edge_case(model, test_case)
            
            result = {
                'case_id': i,
                'type': test_case['type'],
                'description': test_case['description'],
                'evaluation': evaluation
            }
            
            # Add specific information based on case type
            if 'param_name' in test_case:
                result['param_name'] = test_case['param_name']
                result['test_value'] = test_case['test_value']
                
            results.append(result)
            
        return results
        
    def analyze_failure_modes(self, results):
        """Analyze failure modes and patterns"""
        print("Analyzing failure modes...")
        
        # Group results by type
        results_by_type = {}
        for result in results:
            case_type = result['type']
            if case_type not in results_by_type:
                results_by_type[case_type] = []
            results_by_type[case_type].append(result)
            
        # Analyze each type
        failure_analysis = {}
        
        for case_type, type_results in results_by_type.items():
            maes = [r['evaluation']['mae'] for r in type_results]
            mses = [r['evaluation']['mse'] for r in type_results]
            max_errors = [r['evaluation']['max_error'] for r in type_results]
            
            # Find worst cases
            worst_case_idx = np.argmax(maes)
            worst_case = type_results[worst_case_idx]
            
            # Parameter-specific analysis for boundary cases
            param_analysis = {}
            if case_type == 'boundary':
                for result in type_results:
                    param_name = result.get('param_name', 'unknown')
                    if param_name not in param_analysis:
                        param_analysis[param_name] = []
                    param_analysis[param_name].append(result['evaluation']['mae'])
                    
                # Find most problematic parameters
                param_avg_errors = {
                    param: np.mean(errors) for param, errors in param_analysis.items()
                }
                
            failure_analysis[case_type] = {
                'count': len(type_results),
                'mae_stats': {
                    'mean': np.mean(maes),
                    'std': np.std(maes),
                    'min': np.min(maes),
                    'max': np.max(maes),
                    'median': np.median(maes)
                },
                'mse_stats': {
                    'mean': np.mean(mses),
                    'std': np.std(mses),
                    'min': np.min(mses),
                    'max': np.max(mses),
                    'median': np.median(mses)
                },
                'worst_case': {
                    'description': worst_case['description'],
                    'mae': worst_case['evaluation']['mae'],
                    'max_error': worst_case['evaluation']['max_error']
                },
                'param_analysis': param_avg_errors if case_type == 'boundary' else {}
            }
            
        return failure_analysis
        
    def create_edge_case_report(self, results, failure_analysis, output_dir):
        """Create comprehensive edge case evaluation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("EDGE CASE EVALUATION REPORT")
        print("="*80)
        
        # Overall statistics
        all_maes = [r['evaluation']['mae'] for r in results]
        all_mses = [r['evaluation']['mse'] for r in results]
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Total edge cases tested: {len(results)}")
        print(f"Mean Absolute Error - Mean: {np.mean(all_maes):.6f}, Std: {np.std(all_maes):.6f}")
        print(f"Mean Squared Error - Mean: {np.mean(all_mses):.6f}, Std: {np.std(all_mses):.6f}")
        
        # Performance by case type
        print(f"\nPERFORMANCE BY CASE TYPE:")
        for case_type, analysis in failure_analysis.items():
            print(f"\n{case_type.upper()}:")
            print(f"  Cases: {analysis['count']}")
            print(f"  MAE: {analysis['mae_stats']['mean']:.6f} ± {analysis['mae_stats']['std']:.6f}")
            print(f"  Worst case: {analysis['worst_case']['description']} (MAE: {analysis['worst_case']['mae']:.6f})")
            
            # Parameter-specific issues for boundary cases
            if analysis['param_analysis']:
                worst_params = sorted(analysis['param_analysis'].items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Most problematic parameters:")
                for param, avg_error in worst_params:
                    print(f"    {param}: {avg_error:.6f}")
                    
        # Identify systematic issues
        print(f"\nSYSTEMATIC ISSUES:")
        
        # Find parameters that consistently have high errors
        param_error_counts = {}
        for result in results:
            for prob_param in result['evaluation']['problematic_params']:
                param_name = prob_param['name']
                if param_name not in param_error_counts:
                    param_error_counts[param_name] = 0
                param_error_counts[param_name] += 1
                
        if param_error_counts:
            worst_params = sorted(param_error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Parameters with frequent high errors:")
            for param, count in worst_params:
                print(f"  {param}: {count}/{len(results)} cases ({count/len(results)*100:.1f}%)")
        else:
            print("No parameters show consistent high error patterns.")
            
        # Create visualizations
        self.create_edge_case_visualizations(results, failure_analysis, output_dir)
        
        # Save detailed results
        detailed_results = {
            'results': results,
            'failure_analysis': failure_analysis,
            'summary': {
                'total_cases': len(results),
                'overall_mae_mean': np.mean(all_maes),
                'overall_mae_std': np.std(all_maes),
                'overall_mse_mean': np.mean(all_mses),
                'overall_mse_std': np.std(all_mses),
                'problematic_params': dict(sorted(param_error_counts.items(), key=lambda x: x[1], reverse=True))
            }
        }
        
        with open(output_dir / 'edge_case_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
            
        # Create summary CSV
        summary_data = []
        for result in results:
            summary_data.append({
                'case_id': result['case_id'],
                'type': result['type'],
                'description': result['description'],
                'mae': result['evaluation']['mae'],
                'mse': result['evaluation']['mse'],
                'max_error': result['evaluation']['max_error'],
                'num_problematic_params': len(result['evaluation']['problematic_params'])
            })
            
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'edge_case_summary.csv', index=False)
        
        return detailed_results
        
    def create_edge_case_visualizations(self, results, failure_analysis, output_dir):
        """Create visualizations for edge case analysis"""
        plt.figure(figsize=(15, 12))
        
        # 1. Error distribution by case type
        plt.subplot(2, 3, 1)
        case_types = list(failure_analysis.keys())
        type_maes = [failure_analysis[ct]['mae_stats']['mean'] for ct in case_types]
        type_stds = [failure_analysis[ct]['mae_stats']['std'] for ct in case_types]
        
        plt.bar(range(len(case_types)), type_maes, yerr=type_stds, alpha=0.7, capsize=5)
        plt.xticks(range(len(case_types)), case_types, rotation=45)
        plt.ylabel('Mean Absolute Error')
        plt.title('Error by Case Type')
        
        # 2. Error distribution histogram
        plt.subplot(2, 3, 2)
        all_maes = [r['evaluation']['mae'] for r in results]
        plt.hist(all_maes, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Mean Absolute Error')
        plt.ylabel('Count')
        plt.title('Error Distribution')
        
        # 3. Max error vs MAE scatter
        plt.subplot(2, 3, 3)
        all_max_errors = [r['evaluation']['max_error'] for r in results]
        colors = [hash(r['type']) % 7 for r in results]
        plt.scatter(all_maes, all_max_errors, c=colors, alpha=0.6)
        plt.xlabel('Mean Absolute Error')
        plt.ylabel('Maximum Parameter Error')
        plt.title('MAE vs Max Error')
        
        # 4. Parameter error frequency
        plt.subplot(2, 3, 4)
        param_error_counts = {}
        for result in results:
            for prob_param in result['evaluation']['problematic_params']:
                param_name = prob_param['name']
                if param_name not in param_error_counts:
                    param_error_counts[param_name] = 0
                param_error_counts[param_name] += 1
                
        if param_error_counts:
            params = list(param_error_counts.keys())[:10]  # Top 10
            counts = [param_error_counts[p] for p in params]
            plt.barh(range(len(params)), counts, alpha=0.7)
            plt.yticks(range(len(params)), params)
            plt.xlabel('Error Frequency')
            plt.title('Most Problematic Parameters')
        else:
            plt.text(0.5, 0.5, 'No problematic parameters', ha='center', va='center')
            plt.title('Most Problematic Parameters')
            
        # 5. Case type distribution
        plt.subplot(2, 3, 5)
        type_counts = [failure_analysis[ct]['count'] for ct in case_types]
        plt.pie(type_counts, labels=case_types, autopct='%1.1f%%')
        plt.title('Case Type Distribution')
        
        # 6. Error trends
        plt.subplot(2, 3, 6)
        case_ids = [r['case_id'] for r in results]
        plt.plot(case_ids, all_maes, alpha=0.7, linewidth=1)
        plt.xlabel('Case ID')
        plt.ylabel('Mean Absolute Error')
        plt.title('Error Trends')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'edge_case_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Edge case visualizations saved to {output_dir}")
        
    def run_evaluation(self, output_dir='outputs_optimized/edge_case_analysis'):
        """Run the complete edge case evaluation"""
        print("Starting Edge Case Evaluation...")
        
        # Load model
        model = self.load_model()
        
        # Get dataset statistics
        param_stats, dataset_params = self.get_dataset_statistics()
        
        # Generate test cases
        print("Generating test cases...")
        boundary_cases = self.generate_boundary_cases(param_stats)
        extreme_cases = self.generate_extreme_combinations(param_stats, num_combinations=50)
        adversarial_cases = self.generate_adversarial_cases(param_stats, dataset_params, num_cases=30)
        
        all_test_cases = boundary_cases + extreme_cases + adversarial_cases
        print(f"Generated {len(all_test_cases)} total test cases")
        
        # Run evaluation
        results = self.run_edge_case_evaluation(model, all_test_cases)
        
        # Analyze failure modes
        failure_analysis = self.analyze_failure_modes(results)
        
        # Create report
        detailed_results = self.create_edge_case_report(results, failure_analysis, output_dir)
        
        print(f"\nEdge case evaluation complete! Results saved to {output_dir}")
        return detailed_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance on edge cases')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, default='datasets', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs_optimized/edge_case_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    evaluator = EdgeCaseEvaluator(
        model_path=args.model_path,
        dataset_path=args.dataset_path
    )
    
    results = evaluator.run_evaluation(args.output_dir)
    
if __name__ == '__main__':
    main()