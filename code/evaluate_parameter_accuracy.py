#!/usr/bin/env python3
"""
Parameter Accuracy Evaluation Tool

This tool evaluates how well the trained model predicts individual PolyMAX parameters.
It provides detailed per-parameter analysis including MAE, MSE, correlation, and error distribution.
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import argparse

# Import model and data utilities
import sys
sys.path.append('.')
from models.vae.ae import RegressionAE
from utils.data import load_dataset

class ParameterAccuracyEvaluator:
    def __init__(self, model_path, dataset_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.snap_discrete = True  # Heuristic snapping for discrete params
        
        # Load optional parameter schema (for reporting)
        self.param_schema = None
        self.param_names = None
        try:
            with open('params_schema.json', 'r') as f:
                self.param_schema = json.load(f)
            if isinstance(self.param_schema, dict) and 'parameter_order' in self.param_schema:
                self.param_names = self.param_schema['parameter_order']
        except Exception:
            self.param_schema = None
            self.param_names = None
        # num_params will be set after loading dataset if not from schema
        self.num_params = None
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}")
        
        # Load the complete trained model (RegressionAE)
        model = torch.load(self.model_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
        
    def get_predictions(self, model, dataloader):
        """Get model predictions and ground truth for all test samples"""
        all_predictions = []
        all_targets = []
        
        print("Generating predictions...")
        with torch.no_grad():
            for batch_idx, (data, params, meta, audio) in enumerate(dataloader):
                data, params = data.to(self.device), params.to(self.device)
                
                # Get model prediction (parameter predictions)
                p_tilde = model(data)
                # Clamp predictions to valid [0, 1] domain
                p_tilde = torch.clamp(p_tilde, 0.0, 1.0)
                
                all_predictions.append(p_tilde.cpu().numpy())
                all_targets.append(params.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Check for NaN values
        nan_mask_pred = np.isnan(predictions)
        nan_mask_target = np.isnan(targets)
        
        if np.any(nan_mask_pred):
            print(f"Warning: Found {np.sum(nan_mask_pred)} NaN values in predictions")
            predictions = np.nan_to_num(predictions, nan=0.0)
            
        if np.any(nan_mask_target):
            print(f"Warning: Found {np.sum(nan_mask_target)} NaN values in targets")
            targets = np.nan_to_num(targets, nan=0.0)
        
        print(f"Generated predictions for {predictions.shape[0]} samples")
        return predictions, targets

    def snap_discrete_params_if_applicable(self, predictions, targets):
        """Heuristically snap predictions to discrete sets if target shows discreteness.

        Discrete heuristic: parameter has <= 10 unique target values across dataset.
        """
        if not self.snap_discrete:
            return predictions
        try:
            preds = predictions.copy()
            # Unique values per parameter from targets
            for i in range(targets.shape[1]):
                uniq = np.unique(targets[:, i])
                if uniq.shape[0] <= 10:
                    # Snap each prediction to closest unique value
                    # Vectorized nearest neighbor on 1D set
                    # Expand dims: preds_col [N], uniq [K] -> distances [N,K]
                    diffs = np.abs(preds[:, i:i+1] - uniq.reshape(1, -1))
                    nearest = diffs.argmin(axis=1)
                    preds[:, i] = uniq[nearest]
            return preds
        except Exception:
            return predictions
        
    def calculate_per_parameter_metrics(self, predictions, targets):
        """Calculate detailed metrics for each parameter"""
        metrics = {}
        
        for i, param_name in enumerate(self.param_names):
            pred_param = predictions[:, i]
            true_param = targets[:, i]
            
            # Basic metrics
            mae = mean_absolute_error(true_param, pred_param)
            mse = mean_squared_error(true_param, pred_param)
            rmse = np.sqrt(mse)
            # Robust R²: if constant target, set to NaN
            try:
                r2 = r2_score(true_param, pred_param)
            except Exception:
                r2 = np.nan
            
            # Correlation
            try:
                correlation, p_value = pearsonr(true_param, pred_param)
            except Exception:
                correlation, p_value = (np.nan, np.nan)
            
            # Error statistics
            errors = pred_param - true_param
            error_std = np.std(errors)
            error_mean = np.mean(errors)
            
            # Parameter range and relative error
            param_range = np.max(true_param) - np.min(true_param)
            relative_mae = mae / param_range if param_range > 0 else np.nan
            
            metrics[param_name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'correlation': correlation,
                'correlation_p_value': p_value,
                'error_mean': error_mean,
                'error_std': error_std,
                'param_range': param_range,
                'relative_mae': relative_mae,
                'param_min': np.min(true_param),
                'param_max': np.max(true_param),
                'pred_min': np.min(pred_param),
                'pred_max': np.max(pred_param)
            }
            
        return metrics
        
    def create_summary_report(self, metrics):
        """Create a summary report of parameter accuracy"""
        df = pd.DataFrame(metrics).T
        # Replace inf with NaN for robust aggregation
        df = df.replace([np.inf, -np.inf], np.nan)
        
        print("\n" + "="*80)
        print("PARAMETER ACCURACY SUMMARY REPORT")
        print("="*80)
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        print(f"Average MAE: {np.nanmean(df['mae']):.6f}")
        print(f"Average RMSE: {np.nanmean(df['rmse']):.6f}")
        print(f"Average R²: {np.nanmean(df['r2']):.6f}")
        print(f"Average Correlation: {np.nanmean(df['correlation']):.6f}")
        
        # Best performing parameters
        print(f"\nBEST PERFORMING PARAMETERS (by R²):")
        best_params = df.dropna(subset=['r2']).nlargest(5, 'r2')
        for param, row in best_params.iterrows():
            print(f"  {param}: R²={row['r2']:.4f}, MAE={row['mae']:.6f}, Corr={row['correlation']:.4f}")
            
        # Worst performing parameters
        print(f"\nWORST PERFORMING PARAMETERS (by R²):")
        worst_params = df.dropna(subset=['r2']).nsmallest(5, 'r2')
        for param, row in worst_params.iterrows():
            print(f"  {param}: R²={row['r2']:.4f}, MAE={row['mae']:.6f}, Corr={row['correlation']:.4f}")
            
        # Parameters with highest relative error
        print(f"\nHIGHEST RELATIVE ERROR PARAMETERS:")
        high_rel_error = df.dropna(subset=['relative_mae']).nlargest(5, 'relative_mae')
        for param, row in high_rel_error.iterrows():
            print(f"  {param}: Rel_MAE={row['relative_mae']:.4f}, Range={row['param_range']:.6f}")
            
        return df
        
    def create_visualizations(self, metrics, predictions, targets, output_dir):
        """Create visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        df = pd.DataFrame(metrics).T
        
        # 1. Parameter accuracy heatmap
        plt.figure(figsize=(15, 10))
        
        # Create a matrix for heatmap
        heatmap_data = df[['mae', 'rmse', 'r2', 'correlation']].T
        
        sns.heatmap(heatmap_data, annot=False, cmap='RdYlBu_r', center=0)
        plt.title('Parameter Accuracy Heatmap')
        plt.xlabel('Parameters')
        plt.ylabel('Metrics')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. R² distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(df['r2'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('R² Score')
        plt.ylabel('Number of Parameters')
        plt.title('Distribution of R² Scores')
        plt.axvline(df['r2'].mean(), color='red', linestyle='--', label=f'Mean: {df["r2"].mean():.3f}')
        plt.legend()
        
        # 3. MAE vs Parameter Range
        plt.subplot(1, 2, 2)
        plt.scatter(df['param_range'], df['mae'], alpha=0.6)
        plt.xlabel('Parameter Range')
        plt.ylabel('Mean Absolute Error')
        plt.title('MAE vs Parameter Range')
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Top 10 best/worst parameters
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Best parameters
        best_10 = df.nlargest(10, 'r2')
        ax1.barh(range(len(best_10)), best_10['r2'], color='green', alpha=0.7)
        ax1.set_yticks(range(len(best_10)))
        ax1.set_yticklabels(best_10.index)
        ax1.set_xlabel('R² Score')
        ax1.set_title('Top 10 Best Predicted Parameters')
        ax1.grid(True, alpha=0.3)
        
        # Worst parameters
        worst_10 = df.nsmallest(10, 'r2')
        ax2.barh(range(len(worst_10)), worst_10['r2'], color='red', alpha=0.7)
        ax2.set_yticks(range(len(worst_10)))
        ax2.set_yticklabels(worst_10.index)
        ax2.set_xlabel('R² Score')
        ax2.set_title('Top 10 Worst Predicted Parameters')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'best_worst_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
        
    def save_detailed_results(self, metrics, output_dir):
        """Save detailed results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj
        
        serializable_metrics = convert_numpy_types(metrics)
        
        # Save metrics as JSON
        with open(output_dir / 'parameter_metrics.json', 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
        # Save as CSV for easy analysis
        df = pd.DataFrame(metrics).T
        df.to_csv(output_dir / 'parameter_metrics.csv')
        
        print(f"Detailed results saved to {output_dir}")
        
    def run_evaluation(self, output_dir='outputs_optimized/parameter_analysis'):
        """Run the complete parameter accuracy evaluation"""
        print("Starting Parameter Accuracy Evaluation...")
        
        # Load model
        model = self.load_model()
        
        # Create args for dataset loading
        args = argparse.Namespace(
            path='datasets',
            dataset='polymax_dataset',
            data='mel',
            batch_size=32,
            nbworkers=0,
            train_type='fixed'
        )
        
        # Load dataset
        _, _, test_loader, _ = load_dataset(args)
        
        # Use dataset parameter names if schema not present
        if self.param_names is None:
            self.param_names = list(test_loader.dataset.final_params)
        self.num_params = len(self.param_names)
        print(f"Evaluating {self.num_params} parameters: {self.param_names[:5]}...")
        
        # Get predictions
        predictions, targets = self.get_predictions(model, test_loader)

        # Optionally snap discrete parameters to nearest valid target value
        predictions = self.snap_discrete_params_if_applicable(predictions, targets)
        
        # Calculate metrics
        metrics = self.calculate_per_parameter_metrics(predictions, targets)
        
        # Create summary report
        summary_df = self.create_summary_report(metrics)
        
        # Create visualizations
        self.create_visualizations(metrics, predictions, targets, output_dir)
        
        # Save detailed results
        self.save_detailed_results(metrics, output_dir)
        
        print(f"\nEvaluation complete! Results saved to {output_dir}")
        return metrics, summary_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate parameter prediction accuracy')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, default='datasets', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs_optimized/parameter_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    evaluator = ParameterAccuracyEvaluator(
        model_path=args.model_path,
        dataset_path=args.dataset_path
    )
    
    metrics, summary_df = evaluator.run_evaluation(args.output_dir)
    
if __name__ == '__main__':
    main()
