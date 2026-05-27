#!/usr/bin/env python3
"""
Create hold-out train/test splits for XES logs.

"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass
class SplitResult:
    input_file: str
    train_file: str
    test_file: str
    total_traces: int
    train_traces: int
    test_traces: int
    split_strategy: str
    train_ratio: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/test hold-out splits from a XES file")
    parser.add_argument("--input", required=True, help="Path to input .xes file")
    parser.add_argument("--output-dir", required=True, help="Directory for split output files")
    parser.add_argument("--prefix", default="dataset", help="Prefix for output filenames")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio in (0,1), default=0.8")
    parser.add_argument(
        "--strategy",
        choices=["random", "temporal"],
        default="random",
        help="Split strategy: random (seeded) or temporal (by earliest event timestamp)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for random split")
    return parser.parse_args()


def get_namespace(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.index("}")]
    return ""


def qname(namespace: str, local: str) -> str:
    if namespace:
        return f"{{{namespace}}}{local}"
    return local


def first_event_timestamp(trace: ET.Element, namespace: str) -> datetime:
    event_tag = qname(namespace, "event")
    date_tag = qname(namespace, "date")

    for event in trace.findall(event_tag):
        for date_field in event.findall(date_tag):
            if date_field.get("key") == "time:timestamp":
                raw = date_field.get("value", "")
                if not raw:
                    continue
                # Handles both Z and +00:00 suffixes.
                value = raw.replace("Z", "+00:00")
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    continue
    # Missing timestamps are pushed to the end in temporal splits.
    return datetime.max


def split_indices(num_traces: int, train_ratio: float, strategy: str, seed: int, traces: list[ET.Element], namespace: str) -> tuple[list[int], list[int]]:
    indices = list(range(num_traces))

    if strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(indices)
    else:
        indices.sort(key=lambda idx: first_event_timestamp(traces[idx], namespace))

    train_count = int(round(num_traces * train_ratio))
    train_count = max(1, min(train_count, num_traces - 1)) if num_traces > 1 else num_traces

    train_idx = sorted(indices[:train_count])
    test_idx = sorted(indices[train_count:])
    return train_idx, test_idx


def build_split_root(root: ET.Element, trace_indices: list[int], traces: list[ET.Element], namespace: str) -> ET.Element:
    trace_tag = qname(namespace, "trace")
    new_root = ET.Element(root.tag, dict(root.attrib))

    for child in list(root):
        if child.tag != trace_tag:
            new_root.append(copy.deepcopy(child))

    for idx in trace_indices:
        new_root.append(copy.deepcopy(traces[idx]))

    return new_root


def write_xes(path: Path, root: ET.Element, namespace: str) -> None:
    if namespace:
        ET.register_namespace("", namespace)
    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def run_split(input_file: Path, output_dir: Path, prefix: str, train_ratio: float, strategy: str, seed: int) -> SplitResult:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")

    tree = ET.parse(input_file)
    root = tree.getroot()
    namespace = get_namespace(root.tag)
    trace_tag = qname(namespace, "trace")
    traces = [child for child in list(root) if child.tag == trace_tag]

    if len(traces) < 2:
        raise ValueError(f"Need at least 2 traces to split, found {len(traces)} in {input_file}")

    train_idx, test_idx = split_indices(
        num_traces=len(traces),
        train_ratio=train_ratio,
        strategy=strategy,
        seed=seed,
        traces=traces,
        namespace=namespace,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / f"{prefix}_train.xes"
    test_file = output_dir / f"{prefix}_test.xes"
    report_file = output_dir / f"{prefix}_split_report.json"

    train_root = build_split_root(root, train_idx, traces, namespace)
    test_root = build_split_root(root, test_idx, traces, namespace)

    write_xes(train_file, train_root, namespace)
    write_xes(test_file, test_root, namespace)

    result = SplitResult(
        input_file=str(input_file),
        train_file=str(train_file),
        test_file=str(test_file),
        total_traces=len(traces),
        train_traces=len(train_idx),
        test_traces=len(test_idx),
        split_strategy=strategy,
        train_ratio=train_ratio,
        seed=seed,
    )

    report_file.write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()

    result = run_split(
        input_file=Path(args.input),
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
        train_ratio=args.train_ratio,
        strategy=args.strategy,
        seed=args.seed,
    )

    print("Hold-out split created successfully")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
