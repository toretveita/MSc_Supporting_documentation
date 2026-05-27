"""
Statistical Tests for Model Comparison

"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
from sklearn.metrics import confusion_matrix


class ModelComparisonTests:
    """Statistical tests for comparing ML models."""
    
    @staticmethod
    def paired_t_test(
        results_1: List[float],
        results_2: List[float],
        model_name_1: str = "Model 1",
        model_name_2: str = "Model 2"
    ) -> Dict[str, Any]:
        """
        Paired t-test for comparing models on same CV folds.

        """
        if len(results_1) != len(results_2):
            raise ValueError("Results must have same length (same folds)")
        
        t_stat, p_value = stats.ttest_rel(results_1, results_2)
        
        differences = np.array(results_1) - np.array(results_2)
        mean_diff = np.mean(differences)
        
        # Effect size (Cohen's d for paired samples)
        cohen_d = mean_diff / np.std(differences, ddof=1)
        
        return {
            'test': 'Paired t-test',
            'model_1': model_name_1,
            'model_2': model_name_2,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': bool(p_value < 0.05),
            'significant_at_0.01': bool(p_value < 0.01),
            'significant_at_0.001': bool(p_value < 0.001),
            'mean_difference': float(mean_diff),
            'cohen_d': float(cohen_d),
            'interpretation': ModelComparisonTests._interpret_cohen_d(cohen_d),
            'winner': model_name_1 if mean_diff > 0 else model_name_2 if mean_diff < 0 else "tie"
        }
    
    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray,
        pred_1: np.ndarray,
        pred_2: np.ndarray,
        model_name_1: str = "Model 1",
        model_name_2: str = "Model 2"
    ) -> Dict[str, Any]:
        """
        
        Tests whether the two models make errors in systematically different ways.
        
        """
        # Create contingency table
        both_correct = np.sum((pred_1 == y_true) & (pred_2 == y_true))
        both_wrong = np.sum((pred_1 != y_true) & (pred_2 != y_true))
        model1_only_correct = np.sum((pred_1 == y_true) & (pred_2 != y_true))
        model2_only_correct = np.sum((pred_1 != y_true) & (pred_2 == y_true))
        
        # McNemar statistic with continuity correction
        n_discordant = model1_only_correct + model2_only_correct
        
        if n_discordant == 0:
            # No disagreements, can't test
            return {
                'test': "McNemar's test",
                'model_1': model_name_1,
                'model_2': model_name_2,
                'warning': 'Models always agree - test not applicable',
                'statistic': 0.0,
                'p_value': 1.0,
                'significant_at_0.05': False
            }
        
        statistic = (abs(model1_only_correct - model2_only_correct) - 1)**2 / n_discordant
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return {
            'test': "McNemar's test",
            'model_1': model_name_1,
            'model_2': model_name_2,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant_at_0.05': bool(p_value < 0.05),
            'significant_at_0.01': bool(p_value < 0.01),
            'contingency_table': {
                'both_correct': int(both_correct),
                'both_wrong': int(both_wrong),
                f'{model_name_1}_only_correct': int(model1_only_correct),
                f'{model_name_2}_only_correct': int(model2_only_correct)
            },
            'winner': model_name_1 if model1_only_correct > model2_only_correct 
                     else model_name_2 if model2_only_correct > model1_only_correct 
                     else "tie"
        }
    
    @staticmethod
    def wilcoxon_signed_rank(
        results_1: List[float],
        results_2: List[float],
        model_name_1: str = "Model 1",
        model_name_2: str = "Model 2"
    ) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test.
        
        """
        if len(results_1) != len(results_2):
            raise ValueError("Results must have same length")
        
        statistic, p_value = stats.wilcoxon(results_1, results_2)
        
        mean_diff = np.mean(np.array(results_1) - np.array(results_2))
        
        return {
            'test': 'Wilcoxon signed-rank test',
            'model_1': model_name_1,
            'model_2': model_name_2,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant_at_0.05': bool(p_value < 0.05),
            'significant_at_0.01': bool(p_value < 0.01),
            'mean_difference': float(mean_diff),
            'winner': model_name_1 if mean_diff > 0 else model_name_2 if mean_diff < 0 else "tie"
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        results: List[float],
        n_bootstrap: int = 10000,
        confidence: float = 0.95
    ) -> Dict[str, float]:

        results = np.array(results)
        n = len(results)
        
        # Bootstrap sampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(results, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute percentiles
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return {
            'mean': float(np.mean(results)),
            'std': float(np.std(results)),
            'bootstrap_mean': float(np.mean(bootstrap_means)),
            'bootstrap_std': float(np.std(bootstrap_means)),
            f'ci_{int(confidence*100)}_lower': float(ci_lower),
            f'ci_{int(confidence*100)}_upper': float(ci_upper)
        }
    
    @staticmethod
    def _interpret_cohen_d(d: float) -> str:
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
    
    @staticmethod
    def comprehensive_comparison(
        fold_results_1: List[float],
        fold_results_2: List[float],
        test_predictions_1: np.ndarray = None,
        test_predictions_2: np.ndarray = None,
        test_labels: np.ndarray = None,
        model_name_1: str = "Model 1",
        model_name_2: str = "Model 2"
    ) -> Dict[str, Any]:
        """
        Run statistical comparison between two models.

        """
        results = {
            'paired_t_test': ModelComparisonTests.paired_t_test(
                fold_results_1, fold_results_2, model_name_1, model_name_2
            ),
            'wilcoxon_test': ModelComparisonTests.wilcoxon_signed_rank(
                fold_results_1, fold_results_2, model_name_1, model_name_2
            ),
            'bootstrap_ci_model1': ModelComparisonTests.bootstrap_confidence_interval(
                fold_results_1
            ),
            'bootstrap_ci_model2': ModelComparisonTests.bootstrap_confidence_interval(
                fold_results_2
            )
        }
        
        # Add McNemar's test if predictions provided
        if (test_predictions_1 is not None and 
            test_predictions_2 is not None and 
            test_labels is not None):
            results['mcnemar_test'] = ModelComparisonTests.mcnemar_test(
                test_labels, test_predictions_1, test_predictions_2,
                model_name_1, model_name_2
            )
        
        return results
    
    @staticmethod
    def print_comparison_report(comparison_results: Dict[str, Any]):
        """Print formatted report of statistical comparison."""
        
        print("\n" + "="*80)
        print("STATISTICAL COMPARISON REPORT")
        print("="*80)
        
        # Paired t-test
        if 'paired_t_test' in comparison_results:
            t_test = comparison_results['paired_t_test']
            print(f"\n{t_test['test']}:")
            print(f"  Comparing: {t_test['model_1']} vs {t_test['model_2']}")
            print(f"  t-statistic: {t_test['t_statistic']:.4f}")
            print(f"  p-value: {t_test['p_value']:.6f}")
            print(f"  Mean difference: {t_test['mean_difference']:.4f}")
            print(f"  Cohen's d: {t_test['cohen_d']:.4f} ({t_test['interpretation']})")
            
            if t_test['significant_at_0.001']:
                print(f"  *** Highly significant (p < 0.001)")
            elif t_test['significant_at_0.01']:
                print(f"  ** Very significant (p < 0.01)")
            elif t_test['significant_at_0.05']:
                print(f"  * Significant (p < 0.05)")
            else:
                print(f"  Not significant (p >= 0.05)")
            
        # Wilcoxon test
        if 'wilcoxon_test' in comparison_results:
            wilcox = comparison_results['wilcoxon_test']
            print(f"\n{wilcox['test']}:")
            print(f"  Statistic: {wilcox['statistic']:.4f}")
            print(f"  p-value: {wilcox['p_value']:.6f}")
            if wilcox['significant_at_0.05']:
                print(f"  * Significant (p < 0.05)")
            else:
                print(f"  Not significant (p >= 0.05)")
            
        # McNemar's test
        if 'mcnemar_test' in comparison_results:
            mcnemar = comparison_results['mcnemar_test']
            print(f"\n{mcnemar['test']}:")
            if 'warning' not in mcnemar:
                print(f"  Statistic: {mcnemar['statistic']:.4f}")
                print(f"  p-value: {mcnemar['p_value']:.6f}")
                
                table = mcnemar['contingency_table']
                print(f"\n  Contingency Table:")
                print(f"    Both correct: {table['both_correct']}")
                print(f"    Both wrong: {table['both_wrong']}")
                model1_key = f"{mcnemar['model_1']}_only_correct"
                model2_key = f"{mcnemar['model_2']}_only_correct"
                print(f"    {mcnemar['model_1']} only correct: {table[model1_key]}")
                print(f"    {mcnemar['model_2']} only correct: {table[model2_key]}")
                
                if mcnemar['significant_at_0.05']:
                    print(f"  * Significant (p < 0.05)")
                else:
                    print(f"  Not significant (p >= 0.05)")
            else:
                print(f"  {mcnemar['warning']}")
            
        print("\n" + "="*80)
