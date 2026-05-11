"""
evaluate_weak_labels.py
=======================
Evaluate CBIR retrieval quality with weak labels inferred from image paths.

Outputs:
  - results/quantitative_summary.csv
  - results/quantitative_summary.md
  - results/per_query_metrics.csv

Example:
    python evaluate_weak_labels.py --query-count 200 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from feature_extractors import extract_raw_features


DEFAULT_DB_PATH = os.path.join("database", "database.db")
DEFAULT_FEATURES_PATH = os.path.join("database", "features.npy")
DEFAULT_PCA_PATH = os.path.join("database", "pca_model.pkl")
DEFAULT_RESULTS_DIR = "results"


@dataclass
class QueryResult:
    query_index: int
    query_path: str
    label: str
    total_relevant: int
    p_at_k: Dict[int, float]
    r_at_k: Dict[int, float]
    ap_at_k: Dict[int, float]
    search_ms: float
    extract_ms: float
    pca_ms: float


def parse_k_values(raw: str) -> List[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise ValueError("K list is empty. Example: --k-values 1,5,10")
    if any(v <= 0 for v in vals):
        raise ValueError("All K values must be positive integers.")
    return sorted(set(vals))


def load_paths_from_db(db_path: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT ImageID, ImagePath FROM Images ORDER BY ImageID"
        ).fetchall()
    finally:
        conn.close()
    return [row[1] for row in rows]


def infer_label(image_path: str, mode: str, regex: Optional[str] = None) -> str:
    norm = image_path.replace("\\", "/").strip("/")
    parent = os.path.basename(os.path.dirname(norm))
    stem = os.path.splitext(os.path.basename(norm))[0]

    if mode == "parent":
        return parent if parent else stem
    if mode == "stem_prefix":
        parts = re.split(r"[_\-\s]+", stem)
        return parts[0].lower() if parts and parts[0] else stem.lower()
    if mode == "regex":
        if not regex:
            raise ValueError("mode=regex requires --label-regex")
        m = re.search(regex, norm)
        if not m:
            return "unknown"
        if m.groups():
            return m.group(1).lower()
        return m.group(0).lower()
    raise ValueError(f"Unsupported label mode: {mode}")


def choose_query_indices(
    labels: Sequence[str],
    query_count: int,
    seed: int,
) -> List[int]:
    # Exclude labels that only appear once: they have no relevant sample in gallery.
    counts: Dict[str, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    eligible = [i for i, lb in enumerate(labels) if counts.get(lb, 0) > 1]
    if not eligible:
        raise RuntimeError("No eligible queries (all labels appear only once).")

    rng = random.Random(seed)
    if query_count <= 0 or query_count >= len(eligible):
        chosen = eligible
    else:
        chosen = rng.sample(eligible, k=query_count)
    chosen.sort()
    return chosen


def compute_metrics_for_query(
    query_idx: int,
    query_vec: np.ndarray,
    db_matrix: np.ndarray,
    labels: Sequence[str],
    k_values: Sequence[int],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], int, float]:
    t0 = time.perf_counter()
    dists = cdist(query_vec.reshape(1, -1), db_matrix, metric="cosine").flatten()
    search_ms = (time.perf_counter() - t0) * 1000.0

    # Remove the query image itself.
    ranked = np.argsort(dists)
    ranked = ranked[ranked != query_idx]

    query_label = labels[query_idx]
    relevant = np.array([1 if labels[i] == query_label else 0 for i in ranked], dtype=np.int32)
    total_relevant = int(np.sum(relevant))

    p_at_k: Dict[int, float] = {}
    r_at_k: Dict[int, float] = {}
    ap_at_k: Dict[int, float] = {}

    for k in k_values:
        kk = min(k, len(ranked))
        rel_k = relevant[:kk]
        hits = int(np.sum(rel_k))

        p = float(hits / kk) if kk > 0 else 0.0
        r = float(hits / total_relevant) if total_relevant > 0 else 0.0

        if kk > 0 and total_relevant > 0:
            cumsum = np.cumsum(rel_k)
            precisions = cumsum / (np.arange(kk) + 1)
            ap_num = float(np.sum(precisions * rel_k))
            ap_den = float(min(kk, total_relevant))
            ap = ap_num / ap_den if ap_den > 0 else 0.0
        else:
            ap = 0.0

        p_at_k[k] = p
        r_at_k[k] = r
        ap_at_k[k] = ap

    return p_at_k, r_at_k, ap_at_k, total_relevant, search_ms


def aggregate_results(results: Sequence[QueryResult], k_values: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(results)
    out["num_queries"] = float(n)
    if n == 0:
        for k in k_values:
            out[f"P@{k}"] = 0.0
            out[f"R@{k}"] = 0.0
            out[f"mAP@{k}"] = 0.0
        out["search_ms_mean"] = 0.0
        out["extract_ms_mean"] = 0.0
        out["pca_ms_mean"] = 0.0
        out["total_ms_mean"] = 0.0
        return out

    for k in k_values:
        out[f"P@{k}"] = float(np.mean([r.p_at_k[k] for r in results]))
        out[f"R@{k}"] = float(np.mean([r.r_at_k[k] for r in results]))
        out[f"mAP@{k}"] = float(np.mean([r.ap_at_k[k] for r in results]))

    search_ms = np.array([r.search_ms for r in results], dtype=np.float64)
    extract_ms = np.array([r.extract_ms for r in results], dtype=np.float64)
    pca_ms = np.array([r.pca_ms for r in results], dtype=np.float64)

    out["search_ms_mean"] = float(np.mean(search_ms))
    out["extract_ms_mean"] = float(np.mean(extract_ms))
    out["pca_ms_mean"] = float(np.mean(pca_ms))
    out["total_ms_mean"] = float(np.mean(search_ms + extract_ms + pca_ms))
    return out


def write_per_query_csv(path: str, rows: Sequence[QueryResult], k_values: Sequence[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = [
        "query_index",
        "query_path",
        "label",
        "total_relevant",
        "search_ms",
        "extract_ms",
        "pca_ms",
    ]
    for k in k_values:
        headers.extend([f"P@{k}", f"R@{k}", f"AP@{k}"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            d = {
                "query_index": row.query_index,
                "query_path": row.query_path,
                "label": row.label,
                "total_relevant": row.total_relevant,
                "search_ms": f"{row.search_ms:.3f}",
                "extract_ms": f"{row.extract_ms:.3f}",
                "pca_ms": f"{row.pca_ms:.3f}",
            }
            for k in k_values:
                d[f"P@{k}"] = f"{row.p_at_k[k]:.6f}"
                d[f"R@{k}"] = f"{row.r_at_k[k]:.6f}"
                d[f"AP@{k}"] = f"{row.ap_at_k[k]:.6f}"
            writer.writerow(d)


def write_summary_csv(
    path: str,
    summary: Dict[str, float],
    k_values: Sequence[int],
    config: Dict[str, str],
    append: bool = False,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = list(config.keys()) + ["num_queries"]
    for k in k_values:
        headers.extend([f"P@{k}", f"R@{k}", f"mAP@{k}"])
    headers.extend(["search_ms_mean", "extract_ms_mean", "pca_ms_mean", "total_ms_mean"])

    row = {**config}
    for key in headers:
        if key in summary:
            row[key] = f"{summary[key]:.6f}" if key != "num_queries" else str(int(summary[key]))
    for key in ["search_ms_mean", "extract_ms_mean", "pca_ms_mean", "total_ms_mean"]:
        if key in summary:
            row[key] = f"{summary[key]:.3f}"

    mode = "a" if append and os.path.exists(path) else "w"
    need_header = not (append and os.path.exists(path))
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary_markdown_from_csv(md_path: str, csv_path: str) -> None:
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for r in reader:
            rows.append(r)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(r.get(h, "") for h in headers) + " |\n")


def run(args: argparse.Namespace) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, args.db_path)
    features_path = os.path.join(base_dir, args.features_path)
    pca_path = os.path.join(base_dir, args.pca_path)
    results_dir = os.path.join(base_dir, args.results_dir)

    k_values = parse_k_values(args.k_values)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Missing database file: {db_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"Missing PCA model: {pca_path}")

    db_matrix = np.load(features_path).astype(np.float64)
    paths = load_paths_from_db(db_path)
    if db_matrix.shape[0] != len(paths):
        raise RuntimeError(
            f"Mismatch rows: features={db_matrix.shape[0]} vs images={len(paths)}."
        )

    with open(pca_path, "rb") as f:
        pca = pickle.load(f)

    labels = [infer_label(p, args.label_mode, args.label_regex) for p in paths]
    q_indices = choose_query_indices(labels, args.query_count, args.seed)

    print("=" * 72)
    print("WEAK-LABEL CBIR EVALUATION")
    print("=" * 72)
    print(f"Database rows      : {len(paths)}")
    print(f"Query count        : {len(q_indices)}")
    print(f"Label mode         : {args.label_mode}")
    print(f"Query mode         : {args.query_mode}")
    print(f"K values           : {k_values}")
    print(f"Results dir        : {results_dir}")
    print("-" * 72)

    results: List[QueryResult] = []
    for n, q_idx in enumerate(q_indices, 1):
        extract_ms = 0.0
        pca_ms = 0.0

        if args.query_mode == "precomputed":
            q_vec = db_matrix[q_idx]
        elif args.query_mode == "extract":
            abs_path = os.path.join(base_dir, paths[q_idx])
            t0 = time.perf_counter()
            raw = extract_raw_features(abs_path)
            extract_ms = (time.perf_counter() - t0) * 1000.0
            if raw is None:
                print(f"[WARN] Skip query (cannot read): {paths[q_idx]}")
                continue
            t1 = time.perf_counter()
            q_vec = pca.transform(raw.reshape(1, -1)).reshape(-1).astype(np.float64)
            pca_ms = (time.perf_counter() - t1) * 1000.0
        else:
            raise ValueError(f"Unsupported query mode: {args.query_mode}")

        p_at_k, r_at_k, ap_at_k, total_relevant, search_ms = compute_metrics_for_query(
            q_idx, q_vec, db_matrix, labels, k_values
        )
        if total_relevant <= 0:
            continue

        results.append(
            QueryResult(
                query_index=q_idx,
                query_path=paths[q_idx],
                label=labels[q_idx],
                total_relevant=total_relevant,
                p_at_k=p_at_k,
                r_at_k=r_at_k,
                ap_at_k=ap_at_k,
                search_ms=search_ms,
                extract_ms=extract_ms,
                pca_ms=pca_ms,
            )
        )

        if n % max(1, args.progress_every) == 0 or n == len(q_indices):
            print(f"  processed {n}/{len(q_indices)} queries")

    if not results:
        raise RuntimeError("No valid query results computed.")

    summary = aggregate_results(results, k_values)
    config = {
        "config_name": args.config_name,
        "label_mode": args.label_mode,
        "query_mode": args.query_mode,
    }

    per_query_csv = os.path.join(results_dir, "per_query_metrics.csv")
    summary_csv = os.path.join(results_dir, "quantitative_summary.csv")
    summary_md = os.path.join(results_dir, "quantitative_summary.md")

    write_per_query_csv(per_query_csv, results, k_values)
    write_summary_csv(summary_csv, summary, k_values, config, append=args.append_summary)
    write_summary_markdown_from_csv(summary_md, summary_csv)

    print("-" * 72)
    print("SUMMARY")
    for k in k_values:
        print(
            f"  P@{k}: {summary[f'P@{k}']:.4f} | "
            f"R@{k}: {summary[f'R@{k}']:.4f} | "
            f"mAP@{k}: {summary[f'mAP@{k}']:.4f}"
        )
    print(
        "  Avg latency (ms/query): "
        f"extract={summary['extract_ms_mean']:.2f}, "
        f"pca={summary['pca_ms_mean']:.2f}, "
        f"search={summary['search_ms_mean']:.2f}, "
        f"total={summary['total_ms_mean']:.2f}"
    )
    print("-" * 72)
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {per_query_csv}")
    print("=" * 72)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CBIR with weak labels.")
    p.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to SQLite DB.")
    p.add_argument("--features-path", default=DEFAULT_FEATURES_PATH, help="Path to features.npy.")
    p.add_argument("--pca-path", default=DEFAULT_PCA_PATH, help="Path to pca_model.pkl.")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Output directory for metrics.")
    p.add_argument(
        "--label-mode",
        default="parent",
        choices=["parent", "stem_prefix", "regex"],
        help="How to infer weak label from ImagePath.",
    )
    p.add_argument("--label-regex", default=None, help="Regex for label-mode=regex.")
    p.add_argument(
        "--query-mode",
        default="precomputed",
        choices=["precomputed", "extract"],
        help="precomputed: use feature row as query; extract: re-extract query vector from image.",
    )
    p.add_argument("--query-count", type=int, default=100, help="Number of sampled queries (<=0 means all eligible).")
    p.add_argument("--k-values", default="1,5,10,20", help="Comma-separated K values.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for query sampling.")
    p.add_argument("--config-name", default="C0_full", help="Config name written to summary table.")
    p.add_argument("--progress-every", type=int, default=25, help="Print progress every N queries.")
    p.add_argument(
        "--append-summary",
        action="store_true",
        help="Append this run as a new row in quantitative_summary.csv.",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
