"""
CASAS Comparison: Raw vs PM-Prepared Data

Models:
Temporal GNN with data-driven co-occurrence graph
LSTM-only baseline
Random Forest baseline
Gradient Boosting baseline
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


class RawVsPreparedCASASComparison:
    """Compare CASAS prediction quality on raw vs PM-prepared logs."""

    def __init__(
        self,
        raw_xes_path: str,
        prepared_xes_path: str,
        sequence_length: int = 10,
        n_folds: int = 5,
        random_state: int = 42,
        use_checkpoints: bool = True,
        force_retrain: bool = False,
        output_filename: str = "CASAS_ML_results.json",
        output_directory: str = "results",
    ):
        self.raw_xes_path = raw_xes_path
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

    def run(
        self,
        max_traces: int = None,
        gnn_epochs: int = 50,
        lstm_epochs: int = 30,
    ) -> None:
        """Run full raw-vs-prepared evaluation pipeline."""
        print("=" * 80)
        print("CASAS RAW VS PM-PREPARED NEXT-EVENT PREDICTION")
        print("=" * 80)

        raw_results = self._run_variant(
            variant_name="raw",
            xes_path=self.raw_xes_path,
            max_traces=max_traces,
            gnn_epochs=gnn_epochs,
            lstm_epochs=lstm_epochs,
        )

        prepared_results = self._run_variant(
            variant_name="prepared",
            xes_path=self.prepared_xes_path,
            max_traces=max_traces,
            gnn_epochs=gnn_epochs,
            lstm_epochs=lstm_epochs,
        )

        self.results["raw"] = raw_results
        self.results["prepared"] = prepared_results
        self.results["raw_vs_prepared_stats"] = self._compare_variants(raw_results, prepared_results)

        self._print_summary()
        self._save_results()

    def _run_variant(
        self,
        variant_name: str,
        xes_path: str,
        max_traces: int,
        gnn_epochs: int,
        lstm_epochs: int,
    ) -> Dict[str, Any]:
        """Train/evaluate all model families for one dataset variant."""
        print("\n" + "-" * 80)
        print(f"Running variant: {variant_name.upper()} ({xes_path})")
        print("-" * 80)

        checkpoint_file = self.checkpoint_dir / f"casas_{variant_name}_results.json"
        if self.use_checkpoints and not self.force_retrain and checkpoint_file.exists():
            print(f"Loading checkpoint: {checkpoint_file}")
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)

        parser = XESParser(xes_path)
        events_df = parser.parse_xes(max_traces=max_traces)
        sequences, labels, _ = parser.create_sequences(events_df, self.sequence_length)

        num_activities = len(parser.activity_to_idx)
        print(f"Prepared {len(sequences)} sequences with {num_activities} activity classes")

        gnn_results = self._run_temporal_gnn_cv(sequences, labels, num_activities, gnn_epochs)
        baselines_results = self._run_baselines_cv(sequences, labels, num_activities, lstm_epochs)

        variant_results = {
            "dataset_info": {
                "xes_path": xes_path,
                "num_events": int(len(events_df)),
                "num_traces": int(events_df["case_id"].nunique()),
                "num_sequences": int(len(sequences)),
                "num_activities": int(num_activities),
                "sequence_length": int(self.sequence_length),
            },
            "models": {
                "Temporal GNN": gnn_results,
                "LSTM-only": baselines_results["LSTM-only"],
                "Random Forest": baselines_results["Random Forest"],
                "Gradient Boosting": baselines_results["Gradient Boosting"],
            },
        }

        if self.use_checkpoints:
            serializable = self._convert_to_native(variant_results)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            print(f"Saved checkpoint: {checkpoint_file}")

        return self._convert_to_native(variant_results)

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

    def _compare_variants(self, raw_results: Dict[str, Any], prepared_results: Dict[str, Any]) -> Dict[str, Any]:
        """Paired tests per model family: prepared vs raw."""
        print("\n" + "=" * 80)
        print("RAW VS PM-PREPARED STATISTICAL COMPARISON")
        print("=" * 80)

        comparison = {}

        for model_name in raw_results["models"].keys():
            raw_acc = raw_results["models"][model_name]["accuracy"]["values"]
            prep_acc = prepared_results["models"][model_name]["accuracy"]["values"]

            model_comparison = ModelComparisonTests.comprehensive_comparison(
                fold_results_1=prep_acc,
                fold_results_2=raw_acc,
                model_name_1=f"{model_name} (PM-prepared)",
                model_name_2=f"{model_name} (raw)",
            )

            comparison[model_name] = self._convert_to_native(model_comparison)

            paired = model_comparison["paired_t_test"]
            print(
                f"{model_name}: p={paired['p_value']:.6f}, "
                f"winner={paired['winner']}, "
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
                "max": float(np.max(arr)) if len(arr) > 0 else 0.0,
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
        print("FINAL SUMMARY - CASAS RAW VS PM-PREPARED")
        print("=" * 80)

        for variant_name in ["raw", "prepared"]:
            variant = self.results[variant_name]
            info = variant["dataset_info"]
            print(f"\n{variant_name.upper()} dataset:")
            print(
                f"  traces={info['num_traces']}, events={info['num_events']}, "
                f"activities={info['num_activities']}, sequences={info['num_sequences']}"
            )

            for model_name, metrics in variant["models"].items():
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
    default_raw = repo_root / "inputs" / "casas" / "shib010.xes"

    parser = argparse.ArgumentParser(description="CASAS next-event prediction: raw vs PM-prepared comparison")
    parser.add_argument(
        "--raw-xes-path",
        default=str(default_raw),
        help="Path to raw CASAS .xes log (default: repo inputs/casas/shib010.xes)",
    )
    parser.add_argument(
        "--prepared-xes-path",
        default=None,
        help="Path to PM-prepared CASAS .xes log (required for a meaningful comparison)",
    )
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-retrain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-directory", default="results", help="Output directory (default: ML/results)")
    parser.add_argument("--output-filename", default="CASAS_ML_results.json")
    parser.add_argument("--max-traces", type=int, default=None, help="Optional cap on number of traces parsed")
    parser.add_argument("--gnn-epochs", type=int, default=50)
    parser.add_argument("--lstm-epochs", type=int, default=30)
    args = parser.parse_args()

    raw_path = Path(args.raw_xes_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw XES file not found: {raw_path}. "
            f"If you cloned the repo, try: {default_raw} or pass --raw-xes-path"
        )

    if args.prepared_xes_path is None:
        raise ValueError(
            "--prepared-xes-path is required for this script. "
            "Provide your PM-prepared CASAS .xes file."
        )

    prepared_path = Path(args.prepared_xes_path)
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared XES file not found: {prepared_path}")

    comparison = RawVsPreparedCASASComparison(
        raw_xes_path=str(raw_path),
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
