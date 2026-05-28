"""
Architectural Ablation Study for CASAS
=======================================

Compares model architectural components to isolate the contribution of different 
design choices to next-event prediction accuracy. 

"""

import json
import math
import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


import numpy as np
import torch
from sklearn.model_selection import KFold

from models.baseline_models import LSTMOnlyModel, LSTMOnlyTrainer, TraditionalMLBaselines
from helpers.cross_validation import CrossValidationExperiment
from helpers.data_preprocessing import XESParser
from models.gnn_baseline import BaselineGNN, BaselineGNNTrainer, DataDrivenGraphBuilder
from helpers.statistical_tests import ModelComparisonTests


class ArchitecturalAblationCASAS:
    """Compare architectural components on PM-prepared CASAS data."""

    def __init__(
        self,
        prepared_xes_path: str,
        sequence_length: int = 10,
        n_folds: int = 5,
        random_state: int = 42,
        use_checkpoints: bool = True,
        force_retrain: bool = False,
        output_filename: str = "CASAS_ML_architectural_ablation_results.json",
        output_directory: str = "results",
    ):
        self.prepared_xes_path = prepared_xes_path
        self.sequence_length = sequence_length
        self.n_folds = n_folds
        self.random_state = random_state
        self.use_checkpoints = use_checkpoints
        self.force_retrain = force_retrain
        self.output_filename = output_filename
        output_directory_path = Path(output_directory)
        if not output_directory_path.is_absolute():
            output_directory_path = BASE_DIR / output_directory_path
        self.output_directory = output_directory_path

        self.checkpoint_dir = self.output_directory / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Any] = {}
        self.sequences: np.ndarray = None
        self.labels: np.ndarray = None
        self.num_activities: int = 0
        self.edge_index: torch.Tensor = None
        self.edge_weights: torch.Tensor = None

    def run(
        self,
        max_traces: int = None,
        gnn_epochs: int = 50,
        lstm_epochs: int = 30,
    ) -> None:
        """Run full architectural ablation pipeline."""
        print("=" * 80)
        print("CASAS ARCHITECTURAL ABLATION STUDY (PM-PREPARED DATA)")
        print("=" * 80)

        self._load_data(max_traces)
        self._build_graph()

        # Train all architectural variants
        variant_results = {}

        print("\n" + "-" * 80)
        print("VARIANT 1: Temporal GNN (Full) - Graph layers + LSTM")
        print("-" * 80)
        variant_results["Temporal GNN (Full)"] = self._train_temporal_gnn_full(gnn_epochs)

        print("\n" + "-" * 80)
        print("VARIANT 2: GNN without graph layers - Embeddings → LSTM only")
        print("-" * 80)
        variant_results["GNN without Graph"] = self._train_gnn_no_graph(gnn_epochs)

        print("\n" + "-" * 80)
        print("VARIANT 3: LSTM-only (Standard) - 2 layers, dropout=0.3")
        print("-" * 80)
        variant_results["LSTM-only (Standard)"] = self._train_lstm_standard(lstm_epochs)

        print("\n" + "-" * 80)
        print("VARIANT 4: LSTM without dropout - 2 layers, no regularization")
        print("-" * 80)
        variant_results["LSTM no Dropout"] = self._train_lstm_no_dropout(lstm_epochs)

        print("\n" + "-" * 80)
        print("VARIANT 5: LSTM reduced - 1 layer, dropout=0.3")
        print("-" * 80)
        variant_results["LSTM Reduced"] = self._train_lstm_reduced(lstm_epochs)

        print("\n" + "-" * 80)
        print("VARIANT 6 & 7: Random Forest & Gradient Boosting")
        print("-" * 80)
        baseline_results = self._train_tree_baselines()
        variant_results.update(baseline_results)

        self.results["variants"] = variant_results

        # Statistical comparison: all variants vs Temporal GNN (Full) baseline
        self._compare_all_variants(variant_results)

        self._print_summary()
        self._save_results()

    def _load_data(self, max_traces: int) -> None:
        """Load and prepare PM-prepared CASAS sequences."""
        print(f"\nLoading PM-prepared CASAS data from: {self.prepared_xes_path}")

        parser = XESParser(self.prepared_xes_path)
        events_df = parser.parse_xes(max_traces=max_traces)
        self.sequences, self.labels, _ = parser.create_sequences(events_df, self.sequence_length)
        self.num_activities = len(parser.activity_to_idx)

        print(
            f"Loaded {len(self.sequences)} sequences from {events_df['case_id'].nunique()} traces "
            f"with {self.num_activities} activity classes"
        )

    def _build_graph(self) -> None:
        """Build data-driven co-occurrence graph for GNN variants."""
        print("\nBuilding data-driven activity co-occurrence graph...")
        graph_builder = DataDrivenGraphBuilder(min_support=0.01)
        self.edge_index, self.edge_weights = graph_builder.build_graph_from_sequences(
            self.sequences, self.num_activities
        )
        print(f"Graph: {self.edge_index.shape[1]} edges")

    def _train_temporal_gnn_full(self, num_epochs: int) -> Dict[str, Any]:
        """Temporal GNN with full graph + LSTM layers."""
        model_config = {
            "num_activities": self.num_activities,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_gcn_layers": 2,
            "lstm_layers": 2,
            "dropout": 0.3,
        }

        cv_experiment = CrossValidationExperiment(n_splits=self.n_folds, random_state=self.random_state)
        cv_results = cv_experiment.run_cv_comparison(
            sequences=self.sequences,
            labels=self.labels,
            model_configs={
                "Temporal GNN (Full)": {
                    "model": None,
                    "model_class": BaselineGNN,
                    "model_kwargs": model_config,
                    "trainer_class": BaselineGNNTrainer,
                    "trainer_kwargs": {
                        "edge_index": self.edge_index,
                        "edge_weights": self.edge_weights,
                        "learning_rate": 0.001,
                    },
                }
            },
            num_epochs=num_epochs,
            batch_size=32,
            verbose=True,
        )

        return cv_results["Temporal GNN (Full)"]

    def _train_gnn_no_graph(self, num_epochs: int) -> Dict[str, Any]:
        """GNN variant: embeddings directly to LSTM without graph layers (num_gcn_layers=0)."""
        model_config = {
            "num_activities": self.num_activities,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_gcn_layers": 0,  # Key difference: no graph convolution
            "lstm_layers": 2,
            "dropout": 0.3,
        }

        cv_experiment = CrossValidationExperiment(n_splits=self.n_folds, random_state=self.random_state)
        cv_results = cv_experiment.run_cv_comparison(
            sequences=self.sequences,
            labels=self.labels,
            model_configs={
                "GNN without Graph": {
                    "model": None,
                    "model_class": BaselineGNN,
                    "model_kwargs": model_config,
                    "trainer_class": BaselineGNNTrainer,
                    "trainer_kwargs": {
                        "edge_index": self.edge_index,
                        "edge_weights": self.edge_weights,
                        "learning_rate": 0.001,
                    },
                }
            },
            num_epochs=num_epochs,
            batch_size=32,
            verbose=True,
        )

        return cv_results["GNN without Graph"]

    def _train_lstm_standard(self, num_epochs: int) -> Dict[str, Any]:
        """Standard LSTM: 2 layers, dropout=0.3."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.sequences), start=1):
            print(f"  Fold {fold_idx}/{self.n_folds}")

            x_train = self.sequences[train_idx]
            y_train = self.labels[train_idx]
            x_test = self.sequences[test_idx]
            y_test = self.labels[test_idx]

            lstm_model = LSTMOnlyModel(
                num_activities=self.num_activities,
                embedding_dim=64,
                hidden_dim=128,
                num_layers=2,
                dropout=0.3,
            )
            lstm_trainer = LSTMOnlyTrainer(
                model=lstm_model,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                learning_rate=0.001,
            )
            lstm_trainer.train(
                x_train, y_train, x_test, y_test, num_epochs=num_epochs, batch_size=32, verbose=False
            )
            lstm_eval = lstm_trainer.evaluate(x_test, y_test)
            fold_metrics.append(self._extract_fold_metrics(lstm_eval, fold_idx))

        return self._summarize_fold_metrics(fold_metrics)

    def _train_lstm_no_dropout(self, num_epochs: int) -> Dict[str, Any]:
        """LSTM variant: 2 layers, dropout=0.0 (no regularization)."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.sequences), start=1):
            print(f"  Fold {fold_idx}/{self.n_folds}")

            x_train = self.sequences[train_idx]
            y_train = self.labels[train_idx]
            x_test = self.sequences[test_idx]
            y_test = self.labels[test_idx]

            lstm_model = LSTMOnlyModel(
                num_activities=self.num_activities,
                embedding_dim=64,
                hidden_dim=128,
                num_layers=2,
                dropout=0.0,  # Key difference: no dropout
            )
            lstm_trainer = LSTMOnlyTrainer(
                model=lstm_model,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                learning_rate=0.001,
            )
            lstm_trainer.train(
                x_train, y_train, x_test, y_test, num_epochs=num_epochs, batch_size=32, verbose=False
            )
            lstm_eval = lstm_trainer.evaluate(x_test, y_test)
            fold_metrics.append(self._extract_fold_metrics(lstm_eval, fold_idx))

        return self._summarize_fold_metrics(fold_metrics)

    def _train_lstm_reduced(self, num_epochs: int) -> Dict[str, Any]:
        """LSTM variant: 1 layer, dropout=0.3 (reduced complexity)."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.sequences), start=1):
            print(f"  Fold {fold_idx}/{self.n_folds}")

            x_train = self.sequences[train_idx]
            y_train = self.labels[train_idx]
            x_test = self.sequences[test_idx]
            y_test = self.labels[test_idx]

            lstm_model = LSTMOnlyModel(
                num_activities=self.num_activities,
                embedding_dim=64,
                hidden_dim=128,
                num_layers=1,  # Key difference: reduced from 2 to 1
                dropout=0.3,
            )
            lstm_trainer = LSTMOnlyTrainer(
                model=lstm_model,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                learning_rate=0.001,
            )
            lstm_trainer.train(
                x_train, y_train, x_test, y_test, num_epochs=num_epochs, batch_size=32, verbose=False
            )
            lstm_eval = lstm_trainer.evaluate(x_test, y_test)
            fold_metrics.append(self._extract_fold_metrics(lstm_eval, fold_idx))

        return self._summarize_fold_metrics(fold_metrics)

    def _train_tree_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Train Random Forest and Gradient Boosting on PM-prepared data."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_metrics_rf = []
        fold_metrics_gb = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.sequences), start=1):
            print(f"  Fold {fold_idx}/{self.n_folds}")

            x_train = self.sequences[train_idx]
            y_train = self.labels[train_idx]
            x_test = self.sequences[test_idx]
            y_test = self.labels[test_idx]

            rf = TraditionalMLBaselines.train_random_forest(
                x_train, y_train, x_test, y_test, n_estimators=100, random_state=self.random_state
            )
            fold_metrics_rf.append(self._extract_fold_metrics(rf, fold_idx))

            gb = TraditionalMLBaselines.train_gradient_boosting(
                x_train, y_train, x_test, y_test, n_estimators=100, random_state=self.random_state
            )
            fold_metrics_gb.append(self._extract_fold_metrics(gb, fold_idx))

        return {
            "Random Forest": self._summarize_fold_metrics(fold_metrics_rf),
            "Gradient Boosting": self._summarize_fold_metrics(fold_metrics_gb),
        }

    def _compare_all_variants(self, variant_results: Dict[str, Dict[str, Any]]) -> None:
        """Paired t-tests: each variant vs Temporal GNN (Full) baseline."""
        print("\n" + "=" * 80)
        print("ARCHITECTURAL ABLATION: PAIRWISE STATISTICAL COMPARISON")
        print("=" * 80)

        baseline_name = "Temporal GNN (Full)"
        baseline_acc = variant_results[baseline_name]["accuracy"]["values"]

        comparison_results = {}

        for variant_name in variant_results.keys():
            if variant_name == baseline_name:
                continue

            variant_acc = variant_results[variant_name]["accuracy"]["values"]

            model_comparison = ModelComparisonTests.comprehensive_comparison(
                fold_results_1=baseline_acc,
                fold_results_2=variant_acc,
                model_name_1=baseline_name,
                model_name_2=variant_name,
            )

            comparison_results[variant_name] = self._convert_to_native(model_comparison)

            paired = model_comparison["paired_t_test"]
            print(
                f"{variant_name:25s}: p={paired['p_value']:.6f}, "
                f"winner={paired['winner']:20s}, "
                f"Δ={paired['mean_difference']:+.4f}"
            )

        self.results["architectural_comparison_vs_baseline"] = comparison_results

    def _extract_fold_metrics(self, metrics: Dict[str, Any], fold_idx: int) -> Dict[str, float]:
        """Normalize fold metrics format across model families."""
        return {
            "fold": float(fold_idx),
            "test_accuracy": float(metrics.get("accuracy", 0.0)),
            "test_top3": float(metrics.get("top_3_accuracy", 0.0)),
            "test_top5": float(metrics.get("top_5_accuracy", 0.0)),
            "mean_confidence": float(metrics.get("mean_confidence", 0.0)),
        }

    def _summarize_fold_metrics(self, fold_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """Aggregate fold metrics with mean, std, CI."""
        accuracy_values = [f["test_accuracy"] for f in fold_list]
        top3_values = [f["test_top3"] for f in fold_list]
        top5_values = [f["test_top5"] for f in fold_list]
        confidence_values = [f["mean_confidence"] for f in fold_list]

        return {
            "fold_results": fold_list,
            "accuracy": self._compute_aggregate_stats(accuracy_values),
            "top_3_accuracy": self._compute_aggregate_stats(top3_values),
            "top_5_accuracy": self._compute_aggregate_stats(top5_values),
            "mean_confidence": self._compute_aggregate_stats(confidence_values),
        }

    def _compute_aggregate_stats(self, values: List[float]) -> Dict[str, Any]:
        """Compute mean, std, min, max, 95% CI."""
        valid_values = [v for v in values if math.isfinite(v)]

        if not valid_values:
            return {"values": [None] * len(values), "mean": None, "std": None, "min": None, "max": None, "ci_95": None}

        mean = np.mean(valid_values)
        std = np.std(valid_values, ddof=1)
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)

        if len(valid_values) > 1 and math.isfinite(std):
            from scipy import stats

            ci = stats.t.interval(0.95, len(valid_values) - 1, loc=mean, scale=std / np.sqrt(len(valid_values)))
            ci_95 = [float(ci[0]), float(ci[1])] if math.isfinite(ci[0]) and math.isfinite(ci[1]) else None
        else:
            ci_95 = None

        return {
            "values": [float(v) if math.isfinite(v) else None for v in values],
            "mean": float(mean) if math.isfinite(mean) else None,
            "std": float(std) if math.isfinite(std) else None,
            "min": float(min_val) if math.isfinite(min_val) else None,
            "max": float(max_val) if math.isfinite(max_val) else None,
            "ci_95": ci_95,
        }

    def _convert_to_native(self, obj: Any) -> Any:
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.ndarray):
            return [self._convert_to_native(item) for item in obj.tolist()]
        elif isinstance(obj, (np.floating, np.integer)):
            native = obj.item()
            return float(native) if math.isfinite(native) else None
        elif isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_native(item) for item in obj)
        else:
            return obj

    def _print_summary(self) -> None:
        """Print summary of architectural ablation results."""
        print("\n" + "=" * 80)
        print("ARCHITECTURAL ABLATION SUMMARY")
        print("=" * 80)

        for variant_name, results in self.results["variants"].items():
            acc = results["accuracy"]
            ci_95 = acc.get("ci_95")
            if ci_95 is not None:
                ci_str = f"(CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}])"
            else:
                ci_str = "(CI: unavailable)"

            mean_val = acc.get("mean", 0.0)
            std_val = acc.get("std", 0.0)
            print(
                f"{variant_name:25s}: "
                f"accuracy = {mean_val:.4f} ± {std_val:.4f} "
                f"{ci_str}"
            )

    def _save_results(self) -> None:
        """Save results to JSON file."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self.output_directory / self.output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Main execution entrypoint."""

    def find_repo_root(start_dir: Path) -> Path:
        """Walk upwards to find the repository root (contains README.md, inputs/, ML/)."""
        for candidate in [start_dir] + list(start_dir.parents):
            if (
                (candidate / "README.md").exists()
                and (candidate / "inputs").exists()
                and (candidate / "ML").exists()
            ):
                return candidate
        return start_dir.parent

    repo_root = find_repo_root(BASE_DIR)
    default_prepared = repo_root / "inputs" / "casas" / "shib010.xes"

    parser = argparse.ArgumentParser(description="CASAS architectural ablation study (PM-prepared log)")
    parser.add_argument(
        "--prepared-xes-path",
        default=str(default_prepared),
        help="Path to PM-prepared CASAS .xes log (default: repo inputs/casas/shib010.xes)",
    )
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-retrain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-directory", default="results", help="Output directory (default: ML/results)")
    parser.add_argument("--output-filename", default="CASAS_ML_architectural_ablation_results.json")
    parser.add_argument("--max-traces", type=int, default=None, help="Optional cap on number of traces parsed")
    parser.add_argument("--gnn-epochs", type=int, default=50)
    parser.add_argument("--lstm-epochs", type=int, default=30)
    args = parser.parse_args()

    prepared_path = Path(args.prepared_xes_path)
    if not prepared_path.exists():
        raise FileNotFoundError(
            f"Prepared XES file not found: {prepared_path}. "
            f"If you cloned the repo, try: {default_prepared} or pass --prepared-xes-path"
        )

    comparison = ArchitecturalAblationCASAS(
        prepared_xes_path=str(prepared_path),
        sequence_length=args.sequence_length,
        n_folds=args.n_folds,
        random_state=args.random_state,
        use_checkpoints=args.use_checkpoints,
        force_retrain=args.force_retrain,
        output_filename=args.output_filename,
        output_directory=args.output_directory,
    )

    comparison.run(
        max_traces=args.max_traces,
        gnn_epochs=args.gnn_epochs,
        lstm_epochs=args.lstm_epochs,
    )


if __name__ == "__main__":
    main()
