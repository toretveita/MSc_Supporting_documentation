"""
REFIT Prediction Comparison

Models:
Temporal GNN
LSTM
Random Forest
Gradient Boosting
"""

import json
import math
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


class RigorousComparisonREFIT:
    """REFIT next-event prediction comparison across model families."""

    def __init__(
        self,
        xes_path: str,
        sequence_length: int = 5,
        n_folds: int = 5,
        random_state: int = 42,
        use_checkpoints: bool = True,
        force_retrain: bool = False,
        output_filename: str = "REFIT_ML_results.json",
        output_directory: str = "results",
    ):
        self.xes_path = xes_path
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

    def run(
        self,
        max_traces: int = None,
        gnn_epochs: int = 30,
        lstm_epochs: int = 30,
    ) -> None:
        """Run full REFIT evaluation pipeline."""
        print("=" * 80)
        print("REFIT NEXT-EVENT PREDICTION (MODEL FAMILY COMPARISON)")
        print("=" * 80)

        checkpoint_file = self.checkpoint_dir / "refit_model_family_results.json"
        if self.use_checkpoints and not self.force_retrain and checkpoint_file.exists():
            print(f"Loading checkpoint: {checkpoint_file}")
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            self._print_summary()
            return

        parser = XESParser(self.xes_path)
        events_df = parser.parse_xes(max_traces=max_traces)
        sequences, labels, _ = parser.create_sequences(events_df, self.sequence_length)
        num_activities = len(parser.activity_to_idx)

        print(f"Prepared {len(sequences)} sequences with {num_activities} activity classes")

        self.results = {
            "dataset_info": {
                "xes_path": self.xes_path,
                "num_events": int(len(events_df)),
                "num_traces": int(events_df["case_id"].nunique()),
                "num_sequences": int(len(sequences)),
                "num_activities": int(num_activities),
                "sequence_length": int(self.sequence_length),
            },
            "models": {},
        }

        self.results["models"]["Temporal GNN"] = self._run_temporal_gnn_cv(
            sequences, labels, num_activities, gnn_epochs
        )

        baseline_results = self._run_baselines_cv(sequences, labels, num_activities, lstm_epochs)
        self.results["models"]["LSTM-only"] = baseline_results["LSTM-only"]
        self.results["models"]["Random Forest"] = baseline_results["Random Forest"]
        self.results["models"]["Gradient Boosting"] = baseline_results["Gradient Boosting"]

        self.results["pairwise_stats_vs_temporal_gnn"] = self._pairwise_compare_to_temporal_gnn()

        self._print_summary()
        self._save_results()

        if self.use_checkpoints:
            serializable = self._convert_to_native(self.results)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            print(f"Saved checkpoint: {checkpoint_file}")

    def _run_temporal_gnn_cv(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        num_activities: int,
        num_epochs: int,
    ) -> Dict[str, Any]:
        """Cross-validate the temporal co-occurrence GNN."""
        print("\n[Temporal GNN] Running cross-validation...")

        graph_builder = DataDrivenGraphBuilder(min_support=0.01)
        edge_index, edge_weights = graph_builder.build_graph_from_sequences(sequences, num_activities)

        model_config = {
            "num_activities": num_activities,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_gcn_layers": 2,
            "lstm_layers": 2,
            "dropout": 0.3,
        }

        cv_experiment = CrossValidationExperiment(n_splits=self.n_folds, random_state=self.random_state)
        cv_results = cv_experiment.run_cv_comparison(
            sequences=sequences,
            labels=labels,
            model_configs={
                "Temporal GNN": {
                    "model": None,
                    "model_class": BaselineGNN,
                    "model_kwargs": model_config,
                    "trainer_class": BaselineGNNTrainer,
                    "trainer_kwargs": {
                        "edge_index": edge_index,
                        "edge_weights": edge_weights,
                        "learning_rate": 0.001,
                    },
                }
            },
            num_epochs=num_epochs,
            batch_size=32,
            verbose=True,
        )

        return cv_results["Temporal GNN"]

    def _run_baselines_cv(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        num_activities: int,
        lstm_epochs: int,
    ) -> Dict[str, Any]:
        """Cross-validate RF, GB, and LSTM baselines using the same folds."""
        print("\n[Baselines] Running cross-validation...")

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_metrics: Dict[str, List[Dict[str, float]]] = {
            "Random Forest": [],
            "Gradient Boosting": [],
            "LSTM-only": [],
        }

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sequences), start=1):
            print(f"  Fold {fold_idx}/{self.n_folds}")

            x_train = sequences[train_idx]
            y_train = labels[train_idx]
            x_test = sequences[test_idx]
            y_test = labels[test_idx]

            rf = TraditionalMLBaselines.train_random_forest(
                x_train, y_train, x_test, y_test, n_estimators=100, random_state=self.random_state
            )
            fold_metrics["Random Forest"].append(self._extract_fold_metrics(rf, fold_idx))

            gb = TraditionalMLBaselines.train_gradient_boosting(
                x_train, y_train, x_test, y_test, n_estimators=100, random_state=self.random_state
            )
            fold_metrics["Gradient Boosting"].append(self._extract_fold_metrics(gb, fold_idx))

            lstm_model = LSTMOnlyModel(
                num_activities=num_activities,
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
                x_train,
                y_train,
                x_test,
                y_test,
                num_epochs=lstm_epochs,
                batch_size=32,
                verbose=False,
            )
            lstm_eval = lstm_trainer.evaluate(x_test, y_test)
            fold_metrics["LSTM-only"].append(self._extract_fold_metrics(lstm_eval, fold_idx))

        return {
            model_name: self._summarize_fold_metrics(model_folds)
            for model_name, model_folds in fold_metrics.items()
        }

    def _pairwise_compare_to_temporal_gnn(self) -> Dict[str, Any]:
        """Compare Temporal GNN against each baseline with paired tests."""
        print("\n" + "=" * 80)
        print("PAIRWISE SIGNIFICANCE TESTS (TEMPORAL GNN VS BASELINES)")
        print("=" * 80)

        comparison = {}
        gnn_acc = self.results["models"]["Temporal GNN"]["accuracy"]["values"]

        for baseline_name in ["LSTM-only", "Random Forest", "Gradient Boosting"]:
            baseline_acc = self.results["models"][baseline_name]["accuracy"]["values"]
            stats = ModelComparisonTests.comprehensive_comparison(
                fold_results_1=gnn_acc,
                fold_results_2=baseline_acc,
                model_name_1="Temporal GNN",
                model_name_2=baseline_name,
            )
            comparison[baseline_name] = self._convert_to_native(stats)

            paired = stats["paired_t_test"]
            print(
                f"Temporal GNN vs {baseline_name}: "
                f"p={paired['p_value']:.6f}, winner={paired['winner']}, "
                f"mean_diff={paired['mean_difference']:.4f}"
            )

        return comparison

    def _extract_fold_metrics(self, metrics: Dict[str, Any], fold_idx: int) -> Dict[str, float]:
        """Normalize fold metrics format across model families."""
        return {
            "fold": float(fold_idx),
            "test_accuracy": float(metrics.get("accuracy", 0.0)),
            "test_top3": float(metrics.get("top_3_accuracy", 0.0)),
            "test_top5": float(metrics.get("top_5_accuracy", 0.0)),
            "mean_confidence": float(metrics.get("mean_confidence", 0.0)),
        }

    def _summarize_fold_metrics(self, fold_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compute mean/std/CI summaries matching existing result format."""

        def stats_dict(values: List[float]) -> Dict[str, Any]:
            arr = np.array(values, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1))
            sem = float(std / np.sqrt(len(arr))) if len(arr) > 0 else 0.0
            if len(arr) > 1:
                from scipy import stats

                ci_low, ci_high = stats.t.interval(
                    confidence=0.95,
                    df=len(arr) - 1,
                    loc=mean,
                    scale=sem,
                )
            else:
                ci_low, ci_high = mean, mean

            if not np.isfinite(ci_low):
                ci_low = None
            if not np.isfinite(ci_high):
                ci_high = None

            return {
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci_95_lower": float(ci_low) if ci_low is not None else None,
                "ci_95_upper": float(ci_high) if ci_high is not None else None,
                "min": float(np.min(arr)) if len(arr) > 0 else 0.0,
                "max": float(np.max(arr) if len(arr) > 0 else 0.0),
                "values": [float(v) for v in arr.tolist()],
            }

        acc = [f["test_accuracy"] for f in fold_results]
        top3 = [f["test_top3"] for f in fold_results]
        top5 = [f["test_top5"] for f in fold_results]
        conf = [f["mean_confidence"] for f in fold_results]

        return {
            "accuracy": stats_dict(acc),
            "top_3_accuracy": stats_dict(top3),
            "top_5_accuracy": stats_dict(top5),
            "mean_confidence": stats_dict(conf),
            "fold_details": fold_results,
        }

    def _print_summary(self) -> None:
        """Print concise summary."""
        print("\n" + "=" * 80)
        print("FINAL SUMMARY - REFIT MODEL FAMILY COMPARISON")
        print("=" * 80)

        info = self.results["dataset_info"]
        print(
            f"Dataset: traces={info['num_traces']}, events={info['num_events']}, "
            f"activities={info['num_activities']}, sequences={info['num_sequences']}"
        )

        for model_name, metrics in self.results["models"].items():
            acc = metrics["accuracy"]
            print(
                f"  {model_name}: {acc['mean']:.4f} +/- {acc['std']:.4f} "
                f"(95% CI [{acc['ci_95_lower']:.4f}, {acc['ci_95_upper']:.4f}])"
            )

    def _save_results(self) -> None:
        """Persist all results to disk."""
        output_dir = self.output_directory
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / self.output_filename
        serializable = self._convert_to_native(self.results)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        print(f"\nResults saved to {output_file}")

    def _convert_to_native(self, obj: Any) -> Any:
        """Convert numpy values for JSON serialization."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, np.ndarray):
            return [self._convert_to_native(i) for i in obj.tolist()]
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, dict):
            return {
                k: self._convert_to_native(v)
                for k, v in obj.items()
                if k not in ["model", "predictions", "probabilities"]
            }
        if isinstance(obj, list):
            return [self._convert_to_native(i) for i in obj]
        return obj


def main() -> None:
    """Main execution entrypoint."""

    project_root = BASE_DIR.parents[3]

    comparison = RigorousComparisonREFIT(
        xes_path=str(project_root / "Datasets" / "Refit" / "refit_building02.xes"),
        sequence_length=5,
        n_folds=5,
        random_state=42,
        use_checkpoints=True,
        force_retrain=False,
    )

    comparison.run(
        max_traces=None,
        gnn_epochs=30,
        lstm_epochs=30,
    )


if __name__ == "__main__":
    main()
