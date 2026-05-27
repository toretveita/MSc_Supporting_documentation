"""
Cross-Validation for GNN Model Comparison
"""

import numpy as np
import torch
from sklearn.model_selection import KFold
from scipy import stats
from typing import Dict, List, Tuple, Any
import json


class CrossValidationExperiment:
    """Run experiments with proper k-fold cross-validation."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Args:
            n_splits: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def run_cv_comparison(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        model_configs: Dict[str, Dict[str, Any]],
        num_epochs: int = 30,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict[str, Any]:
      
        results = {}
        
        for model_name, config in model_configs.items():
            if verbose:
                print(f"\n{'='*80}")
                print(f"Cross-validating: {model_name}")
                print(f"{'='*80}")
            
            fold_results = self._run_cv_single_model(
                sequences=sequences,
                labels=labels,
                model_config=config,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            
            results[model_name] = self._compute_statistics(fold_results)
            
            if verbose:
                self._print_model_summary(model_name, results[model_name])
        
        # Add statistical comparison if exactly 2 models
        if len(model_configs) == 2:
            results['statistical_comparison'] = self._compare_two_models(results)
            
            if verbose:
                self._print_statistical_comparison(results['statistical_comparison'])
        
        return results
    
    def _run_cv_single_model(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        model_config: Dict[str, Any],
        model_name: str,
        num_epochs: int,
        batch_size: int,
        verbose: bool
    ) -> List[Dict[str, float]]:
        """Run cross-validation for a single model."""
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(sequences), 1):
            if verbose:
                print(f"\n  Fold {fold}/{self.n_splits}...")
            
            # Split data
            train_sequences = sequences[train_idx]
            train_labels = labels[train_idx]
            test_sequences = sequences[test_idx]
            test_labels = labels[test_idx]
            
            # Create model for this fold (fresh instance each time)
            if model_config['model'] is None:
                # Create new model instance
                model = model_config['model_class'](**model_config['model_kwargs'])
            else:
                # Use provided model
                model = model_config['model']
            
            # Train model
            trainer = model_config['trainer_class'](
                model=model,
                **model_config['trainer_kwargs']
            )
            
            # Train
            train_losses, _ = trainer.train(
                train_sequences=train_sequences,
                train_labels=train_labels,
                test_sequences=test_sequences,
                test_labels=test_labels,
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=False  # Don't print each epoch
            )
            
            # Evaluate on test fold
            metrics = trainer.evaluate(test_sequences, test_labels)
            
            fold_results.append({
                'fold': fold,
                'test_accuracy': metrics['accuracy'],
                'test_top3': metrics.get('top_3_accuracy', 0.0),
                'test_top5': metrics.get('top_5_accuracy', 0.0),
                'mean_confidence': metrics.get('mean_confidence', 0.0),
                'final_train_loss': train_losses[-1] if train_losses else 0.0
            })
            
            if verbose:
                print(f"    Accuracy: {metrics['accuracy']:.4f}, "
                      f"Top-3: {metrics.get('top_3_accuracy', 0.0):.4f}")
        
        return fold_results
    
    def _compute_statistics(self, fold_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compute statistics across folds."""
        
        # Extract metrics
        accuracies = [f['test_accuracy'] for f in fold_results]
        top3_accs = [f['test_top3'] for f in fold_results]
        top5_accs = [f['test_top5'] for f in fold_results]
        confidences = [f['mean_confidence'] for f in fold_results]
        
        # Compute statistics
        def stats_dict(values):
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Sample std
            sem = stats.sem(values)  # Standard error
            ci = stats.t.interval(
                confidence=0.95,
                df=len(values)-1,
                loc=mean,
                scale=sem
            )
            
            return {
                'mean': float(mean),
                'std': float(std),
                'sem': float(sem),
                'ci_95_lower': float(ci[0]),
                'ci_95_upper': float(ci[1]),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
        
        return {
            'accuracy': stats_dict(accuracies),
            'top_3_accuracy': stats_dict(top3_accs),
            'top_5_accuracy': stats_dict(top5_accs),
            'mean_confidence': stats_dict(confidences),
            'fold_details': fold_results
        }
    
    def _compare_two_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical comparison between two models."""
        
        model_names = [k for k in results.keys() if k != 'statistical_comparison']
        
        if len(model_names) != 2:
            return {}
        
        model1, model2 = model_names
        
        # Get fold-wise accuracies
        acc1 = results[model1]['accuracy']['values']
        acc2 = results[model2]['accuracy']['values']
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(acc1, acc2)
        
        # Effect size (Cohen's d for paired samples)
        differences = np.array(acc1) - np.array(acc2)
        cohen_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Determine winner
        if p_value < 0.05:
            winner = model1 if np.mean(acc1) > np.mean(acc2) else model2
            significant = True
        else:
            winner = "No significant difference"
            significant = False
        
        return {
            'model_1': model1,
            'model_2': model2,
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_0.05': significant,
                'significant_at_0.01': p_value < 0.01,
                'significant_at_0.001': p_value < 0.001
            },
            'effect_size': {
                'cohen_d': float(cohen_d),
                'interpretation': self._interpret_cohen_d(cohen_d)
            },
            'mean_difference': float(np.mean(differences)),
            'winner': winner
        }
    
    def _interpret_cohen_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _print_model_summary(self, model_name: str, results: Dict[str, Any]):
        """Print summary statistics for a model."""
        acc = results['accuracy']
        
        print(f"\n  Results for {model_name}:")
        print(f"    Accuracy: {acc['mean']:.4f} ± {acc['std']:.4f}")
        print(f"    95% CI: [{acc['ci_95_lower']:.4f}, {acc['ci_95_upper']:.4f}]")
        print(f"    Range: [{acc['min']:.4f}, {acc['max']:.4f}]")
    
    def _print_statistical_comparison(self, comparison: Dict[str, Any]):
        """Print statistical comparison results."""
        if not comparison:
            return
        
        print(f"\n{'='*80}")
        print("STATISTICAL COMPARISON")
        print(f"{'='*80}")
        
        print(f"\nComparing: {comparison['model_1']} vs {comparison['model_2']}")
        
        t_test = comparison['paired_t_test']
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_test['t_statistic']:.4f}")
        print(f"  p-value: {t_test['p_value']:.6f}")
        
        if t_test['significant_at_0.001']:
            print(f"  *** Highly significant (p < 0.001)")
        elif t_test['significant_at_0.01']:
            print(f"  ** Very significant (p < 0.01)")
        elif t_test['significant_at_0.05']:
            print(f"  * Significant (p < 0.05)")
        else:
            print(f"  Not significant (p >= 0.05)")
        
        effect = comparison['effect_size']
        print(f"\nEffect size:")
        print(f"  Cohen's d: {effect['cohen_d']:.4f} ({effect['interpretation']})")
        print(f"  Mean difference: {comparison['mean_difference']:.4f}")
        
        print(f"\nWinner: {comparison['winner']}")
        
        print(f"\n{'='*80}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save cross-validation results to JSON file."""
        
        # Convert numpy types to Python native types
        serializable_results = self._convert_to_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n>> Cross-validation results saved to {filepath}")
