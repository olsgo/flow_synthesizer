#!/usr/bin/env python3
"""
Parameter Space Coverage Analysis Tool

This tool analyzes the coverage of the parameter space in the training dataset by:
1. Identifying underrepresented parameter regions
2. Detecting parameter correlations and dependencies
3. Finding gaps in parameter combinations
4. Analyzing distribution uniformity across parameter ranges
5. Identifying potential biases in the dataset
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import argparse
from collections import defaultdict

# Import data utilities
import sys
sys.path.append('.')
from utils.data import load_dataset

class ParameterCoverageAnalyzer:
    def __init__(self, dataset_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dataset_path = dataset_path
        
        # Load parameter schema
        with open('params_schema.json', 'r') as f:
            self.param_schema = json.load(f)
        
        self.param_names = self.param_schema['parameter_order']
        self.num_params = len(self.param_names)
        
        print(f"Parameter coverage analysis for {self.num_params} parameters")
        
    def load_dataset_parameters(self):
        """Load all parameters from the dataset"""
        print("Loading dataset parameters...")
        
        args = argparse.Namespace(
            path=self.dataset_path,
            dataset='polymax_dataset',
            data='mel',
            batch_size=64,
            nbworkers=4,
            train_type='random'
        )
        
        train_loader, val_loader, test_loader, _ = load_dataset(args)
        
        all_parameters = []
        
        # Collect parameters from all splits
        for loader_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            print(f"Loading {loader_name} parameters...")
            for data, params, meta, audio in loader:
                all_parameters.append(params)
                
        # Concatenate all parameters
        self.parameters = torch.cat(all_parameters, dim=0).numpy()
        print(f"Loaded {self.parameters.shape[0]} parameter sets")
        
        return self.parameters
        
    def analyze_parameter_distributions(self):
        """Analyze the distribution of each parameter"""
        print("Analyzing parameter distributions...")
        
        distributions = {}
        
        for i, param_name in enumerate(self.param_names):
            param_values = self.parameters[:, i]
            
            # Basic statistics
            stats_dict = {
                'mean': np.mean(param_values),
                'std': np.std(param_values),
                'min': np.min(param_values),
                'max': np.max(param_values),
                'median': np.median(param_values),
                'q25': np.percentile(param_values, 25),
                'q75': np.percentile(param_values, 75),
                'range': np.max(param_values) - np.min(param_values)
            }
            
            # Distribution shape analysis
            skewness = stats.skew(param_values)
            kurtosis = stats.kurtosis(param_values)
            
            # Normality test
            _, normality_p = stats.normaltest(param_values)
            
            # Uniformity test (Kolmogorov-Smirnov against uniform distribution)
            uniform_min, uniform_max = np.min(param_values), np.max(param_values)
            uniform_dist = stats.uniform(loc=uniform_min, scale=uniform_max - uniform_min)
            _, uniformity_p = stats.kstest(param_values, uniform_dist.cdf)
            
            # Coverage analysis
            expected_range = self.param_schema.get(param_name, {}).get('range', [0, 1])
            actual_range = [np.min(param_values), np.max(param_values)]
            coverage_ratio = (actual_range[1] - actual_range[0]) / (expected_range[1] - expected_range[0])
            
            # Density analysis (find sparse regions)
            hist, bin_edges = np.histogram(param_values, bins=20)
            sparse_bins = np.where(hist < np.percentile(hist, 10))[0]
            sparse_regions = [(bin_edges[i], bin_edges[i+1]) for i in sparse_bins]
            
            distributions[param_name] = {
                'index': i,
                'statistics': stats_dict,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'normality_p': normality_p,
                'uniformity_p': uniformity_p,
                'is_normal': normality_p > 0.05,
                'is_uniform': uniformity_p > 0.05,
                'expected_range': expected_range,
                'actual_range': actual_range,
                'coverage_ratio': coverage_ratio,
                'sparse_regions': sparse_regions,
                'histogram': {'counts': hist.tolist(), 'bin_edges': bin_edges.tolist()}
            }
            
        return distributions
        
    def analyze_parameter_correlations(self):
        """Analyze correlations between parameters"""
        print("Analyzing parameter correlations...")
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(self.parameters.T)
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(self.param_names)):
            for j in range(i+1, len(self.param_names)):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'param1': self.param_names[i],
                        'param2': self.param_names[j],
                        'correlation': corr,
                        'strength': 'very_strong' if abs(corr) > 0.9 else 'strong'
                    })
                    
        # Identify parameter clusters using correlation
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        return {
            'correlation_matrix': correlation_matrix,
            'strong_correlations': strong_correlations,
            'distance_matrix': distance_matrix
        }
        
    def identify_parameter_clusters(self, n_clusters=5):
        """Identify clusters in parameter space"""
        print(f"Identifying {n_clusters} parameter clusters...")
        
        # Standardize parameters
        scaler = StandardScaler()
        scaled_params = scaler.fit_transform(self.parameters)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_params)
        
        # Analyze clusters
        clusters = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_params = self.parameters[cluster_mask]
            
            # Cluster statistics
            cluster_center = np.mean(cluster_params, axis=0)
            cluster_std = np.std(cluster_params, axis=0)
            cluster_size = np.sum(cluster_mask)
            
            # Find most characteristic parameters for this cluster
            param_importance = cluster_std / (np.std(self.parameters, axis=0) + 1e-8)
            top_params = np.argsort(param_importance)[-3:]  # Top 3 most important
            
            clusters[i] = {
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(self.parameters) * 100),
                'center': cluster_center.tolist(),
                'std': cluster_std.tolist(),
                'top_characteristic_params': [
                    {'name': self.param_names[idx], 'importance': float(param_importance[idx])}
                    for idx in top_params
                ]
            }
            
        return {
            'clusters': clusters,
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'scaler': scaler
        }
        
    def find_coverage_gaps(self, grid_resolution=10):
        """Find gaps in parameter space coverage using grid analysis"""
        print("Finding coverage gaps...")
        
        # For computational efficiency, analyze pairs of most important parameters
        # Use PCA to identify most important parameter combinations
        pca = PCA(n_components=min(10, self.num_params))
        pca_params = pca.fit_transform(self.parameters)
        
        # Analyze coverage in 2D projections of top principal components
        gaps = []
        
        for i in range(min(3, pca_params.shape[1])):  # Top 3 PCs
            for j in range(i+1, min(3, pca_params.shape[1])):
                # Create 2D grid
                x_min, x_max = np.min(pca_params[:, i]), np.max(pca_params[:, i])
                y_min, y_max = np.min(pca_params[:, j]), np.max(pca_params[:, j])
                
                x_grid = np.linspace(x_min, x_max, grid_resolution)
                y_grid = np.linspace(y_min, y_max, grid_resolution)
                
                # Count samples in each grid cell
                grid_counts = np.zeros((grid_resolution, grid_resolution))
                
                for k in range(len(pca_params)):
                    x_idx = min(int((pca_params[k, i] - x_min) / (x_max - x_min) * (grid_resolution - 1)), grid_resolution - 1)
                    y_idx = min(int((pca_params[k, j] - y_min) / (y_max - y_min) * (grid_resolution - 1)), grid_resolution - 1)
                    grid_counts[x_idx, y_idx] += 1
                    
                # Find empty or sparse cells
                sparse_threshold = np.percentile(grid_counts[grid_counts > 0], 10)
                sparse_cells = np.where(grid_counts <= sparse_threshold)
                
                gap_info = {
                    'pc_pair': (i, j),
                    'grid_counts': grid_counts,
                    'sparse_cells': len(sparse_cells[0]),
                    'total_cells': grid_resolution * grid_resolution,
                    'coverage_ratio': len(sparse_cells[0]) / (grid_resolution * grid_resolution),
                    'x_range': (x_min, x_max),
                    'y_range': (y_min, y_max)
                }
                
                gaps.append(gap_info)
                
        return {
            'pca': pca,
            'pca_params': pca_params,
            'gaps': gaps
        }
        
    def analyze_boundary_coverage(self):
        """Analyze coverage near parameter boundaries"""
        print("Analyzing boundary coverage...")
        
        boundary_analysis = {}
        
        for i, param_name in enumerate(self.param_names):
            param_values = self.parameters[:, i]
            expected_range = self.param_schema.get(param_name, {}).get('range', [0, 1])
            
            # Define boundary regions (10% from each end)
            range_size = expected_range[1] - expected_range[0]
            boundary_size = range_size * 0.1
            
            lower_boundary = expected_range[0] + boundary_size
            upper_boundary = expected_range[1] - boundary_size
            
            # Count samples in boundary regions
            lower_boundary_samples = np.sum(param_values <= lower_boundary)
            upper_boundary_samples = np.sum(param_values >= upper_boundary)
            middle_samples = np.sum((param_values > lower_boundary) & (param_values < upper_boundary))
            
            total_samples = len(param_values)
            
            boundary_analysis[param_name] = {
                'expected_range': expected_range,
                'boundary_size': boundary_size,
                'lower_boundary_samples': int(lower_boundary_samples),
                'upper_boundary_samples': int(upper_boundary_samples),
                'middle_samples': int(middle_samples),
                'lower_boundary_ratio': float(lower_boundary_samples / total_samples),
                'upper_boundary_ratio': float(upper_boundary_samples / total_samples),
                'middle_ratio': float(middle_samples / total_samples),
                'boundary_balance': abs(lower_boundary_samples - upper_boundary_samples) / total_samples
            }
            
        return boundary_analysis
        
    def detect_parameter_biases(self):
        """Detect systematic biases in parameter distributions"""
        print("Detecting parameter biases...")
        
        biases = {}
        
        for i, param_name in enumerate(self.param_names):
            param_values = self.parameters[:, i]
            expected_range = self.param_schema.get(param_name, {}).get('range', [0, 1])
            
            # Normalize to [0, 1] for bias analysis
            normalized_values = (param_values - expected_range[0]) / (expected_range[1] - expected_range[0])
            
            # Central tendency bias
            median_bias = abs(np.median(normalized_values) - 0.5)
            mean_bias = abs(np.mean(normalized_values) - 0.5)
            
            # Extreme value bias
            extreme_threshold = 0.1  # 10% from boundaries
            extreme_values = np.sum((normalized_values < extreme_threshold) | (normalized_values > 1 - extreme_threshold))
            extreme_ratio = extreme_values / len(normalized_values)
            
            # Mode analysis
            hist, bin_edges = np.histogram(normalized_values, bins=10)
            mode_bin = np.argmax(hist)
            mode_position = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
            mode_bias = abs(mode_position - 0.5)
            
            # Clustering bias (check if values cluster around specific points)
            # Use coefficient of variation
            cv = np.std(normalized_values) / (np.mean(normalized_values) + 1e-8)
            
            biases[param_name] = {
                'median_bias': float(median_bias),
                'mean_bias': float(mean_bias),
                'extreme_ratio': float(extreme_ratio),
                'mode_bias': float(mode_bias),
                'coefficient_of_variation': float(cv),
                'is_center_biased': median_bias < 0.1,
                'is_extreme_poor': extreme_ratio < 0.1,
                'is_highly_clustered': cv < 0.3
            }
            
        return biases
        
    def create_coverage_report(self, distributions, correlations, clusters, gaps, boundaries, biases, output_dir):
        """Create comprehensive coverage analysis report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("PARAMETER SPACE COVERAGE ANALYSIS REPORT")
        print("="*80)
        
        # Distribution summary
        print(f"\nDISTRIBUTION ANALYSIS:")
        normal_params = [name for name, info in distributions.items() if info['is_normal']]
        uniform_params = [name for name, info in distributions.items() if info['is_uniform']]
        
        print(f"Normal distributions: {len(normal_params)}/{len(distributions)} parameters")
        print(f"Uniform distributions: {len(uniform_params)}/{len(distributions)} parameters")
        
        # Coverage summary
        poor_coverage = [name for name, info in distributions.items() if info['coverage_ratio'] < 0.8]
        print(f"\nCOVERAGE ISSUES:")
        print(f"Poor coverage (<80% of expected range): {len(poor_coverage)} parameters")
        if poor_coverage:
            print(f"Parameters with poor coverage: {', '.join(poor_coverage)}")
            
        # Correlation summary
        print(f"\nCORRELATION ANALYSIS:")
        print(f"Strong correlations found: {len(correlations['strong_correlations'])}")
        for corr in correlations['strong_correlations'][:5]:  # Show top 5
            print(f"  {corr['param1']} ↔ {corr['param2']}: {corr['correlation']:.3f}")
            
        # Cluster summary
        print(f"\nCLUSTER ANALYSIS:")
        cluster_sizes = [info['percentage'] for info in clusters['clusters'].values()]
        print(f"Cluster size range: {min(cluster_sizes):.1f}% - {max(cluster_sizes):.1f}%")
        print(f"Largest cluster: {max(cluster_sizes):.1f}% of data")
        
        # Bias summary
        print(f"\nBIAS ANALYSIS:")
        center_biased = [name for name, info in biases.items() if info['is_center_biased']]
        extreme_poor = [name for name, info in biases.items() if info['is_extreme_poor']]
        highly_clustered = [name for name, info in biases.items() if info['is_highly_clustered']]
        
        print(f"Center-biased parameters: {len(center_biased)}")
        print(f"Poor extreme value coverage: {len(extreme_poor)}")
        print(f"Highly clustered parameters: {len(highly_clustered)}")
        
        # Create visualizations
        self.create_coverage_visualizations(distributions, correlations, clusters, gaps, boundaries, biases, output_dir)
        
        # Save detailed results
        results = {
            'distributions': distributions,
            'correlations': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in correlations.items()},
            'clusters': {k: v for k, v in clusters.items() if k != 'scaler'},
            'boundaries': boundaries,
            'biases': biases,
            'summary': {
                'total_parameters': len(distributions),
                'normal_distributions': len(normal_params),
                'uniform_distributions': len(uniform_params),
                'poor_coverage_count': len(poor_coverage),
                'strong_correlations_count': len(correlations['strong_correlations']),
                'center_biased_count': len(center_biased),
                'extreme_poor_count': len(extreme_poor)
            }
        }
        
        with open(output_dir / 'coverage_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        return results
        
    def create_coverage_visualizations(self, distributions, correlations, clusters, gaps, boundaries, biases, output_dir):
        """Create comprehensive visualizations for coverage analysis"""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Parameter distribution overview
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, (param_name, info) in enumerate(list(distributions.items())[:12]):
            if i < len(axes):
                param_values = self.parameters[:, info['index']]
                axes[i].hist(param_values, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{param_name}\nCoverage: {info["coverage_ratio"]:.2f}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Count')
                
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlations['correlation_matrix'], dtype=bool))
        sns.heatmap(correlations['correlation_matrix'], 
                   mask=mask,
                   xticklabels=self.param_names,
                   yticklabels=self.param_names,
                   annot=False,
                   cmap='RdBu_r',
                   center=0,
                   square=True)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Coverage quality overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Coverage ratios
        coverage_ratios = [info['coverage_ratio'] for info in distributions.values()]
        axes[0, 0].hist(coverage_ratios, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0.8, color='red', linestyle='--', label='80% threshold')
        axes[0, 0].set_title('Coverage Ratio Distribution')
        axes[0, 0].set_xlabel('Coverage Ratio')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        
        # Bias analysis
        median_biases = [info['median_bias'] for info in biases.values()]
        axes[0, 1].hist(median_biases, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Median Bias Distribution')
        axes[0, 1].set_xlabel('Median Bias')
        axes[0, 1].set_ylabel('Count')
        
        # Cluster sizes
        cluster_sizes = [info['percentage'] for info in clusters['clusters'].values()]
        axes[1, 0].bar(range(len(cluster_sizes)), cluster_sizes, alpha=0.7)
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Percentage of Data')
        
        # Boundary coverage
        boundary_ratios = [info['lower_boundary_ratio'] + info['upper_boundary_ratio'] 
                          for info in boundaries.values()]
        axes[1, 1].hist(boundary_ratios, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Boundary Coverage Distribution')
        axes[1, 1].set_xlabel('Boundary Coverage Ratio')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'coverage_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. PCA visualization
        if 'pca_params' in gaps:
            plt.figure(figsize=(12, 5))
            
            # First two principal components
            plt.subplot(1, 2, 1)
            plt.scatter(gaps['pca_params'][:, 0], gaps['pca_params'][:, 1], alpha=0.6, s=1)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('Parameter Space in PCA Coordinates')
            
            # Explained variance
            plt.subplot(1, 2, 2)
            explained_var = gaps['pca'].explained_variance_ratio_
            plt.bar(range(len(explained_var)), explained_var)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Explained Variance')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"Coverage visualizations saved to {output_dir}")
        
    def run_analysis(self, output_dir='outputs_optimized/coverage_analysis'):
        """Run the complete parameter space coverage analysis"""
        print("Starting Parameter Space Coverage Analysis...")
        
        # Load dataset
        parameters = self.load_dataset_parameters()
        
        # Run all analyses
        distributions = self.analyze_parameter_distributions()
        correlations = self.analyze_parameter_correlations()
        clusters = self.identify_parameter_clusters()
        gaps = self.find_coverage_gaps()
        boundaries = self.analyze_boundary_coverage()
        biases = self.detect_parameter_biases()
        
        # Create comprehensive report
        results = self.create_coverage_report(
            distributions, correlations, clusters, gaps, boundaries, biases, output_dir
        )
        
        print(f"\nCoverage analysis complete! Results saved to {output_dir}")
        return results

def main():
    parser = argparse.ArgumentParser(description='Analyze parameter space coverage')
    parser.add_argument('--dataset_path', type=str, default='datasets', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs_optimized/coverage_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = ParameterCoverageAnalyzer(dataset_path=args.dataset_path)
    results = analyzer.run_analysis(args.output_dir)
    
if __name__ == '__main__':
    main()