#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cartography plot generator for ACL classification mapping.

Usage:
    python make_cartography_plots.py \
        --input all_papers_with_classes_and_metrics.csv \
        --outdir cartography_outputs

Notes:
- Expects columns: year, venue, citations, type, domain, is_benchmark,
  num_proposed_benchmarks, n_classes, metric_type, title, abstract.
- If some columns are missing, the script infers reasonable defaults
  (e.g., heuristic 'type', 'domain', 'is_benchmark', 'n_classes', 'metric_type').
- Uses matplotlib (no seaborn). Each figure is a separate PNG.
- Note, that this file was compiled with the help of Claude 4.5 Sonnet Generative AI tool.
"""

import os
import re
import argparse
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def infer_type(text: str) -> str:
    t = str(text).lower()
    if re.search(r"multi[\s-]?label", t):
        return "multi-label"
    if re.search(r"multi[\s-]?class", t) or re.search(r"\bn[- ]?way\b", t):
        return "multi-class"
    if re.search(r"\btwo[\s-]?class\b", t) or "binary" in t or "positive/negative" in t:
        return "binary"
    if "classification" in t or "classifier" in t:
        return "unspecified"
    return "unknown"


DOMAIN_KWS = {
    "sentiment": ["sentiment", "polarity", "opinion"],
    "qa": ["question answering", "qa", "reading comprehension"],
    "ner": ["named entity", "ner", "entity recognition"],
    "intent": ["intent", "utterance", "intent detection"],
    "emotion": ["emotion", "affect", "feeling"],
    "topic": ["topic classification", "topic detection", "news classification"],
    "toxic": ["toxic", "hate speech", "offensive", "abusive"],
    "stance": ["stance detection", "stance classification"],
    "medical": ["medical", "clinical", "biomedical", "health", "depression", "disease"],
    "legal": ["legal", "law", "contract"],
}


def infer_domain(text: str) -> str:
    s = str(text).lower()
    for d, kws in DOMAIN_KWS.items():
        if any(k in s for k in kws):
            return d
    return "general"


def detect_benchmark_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    s = text.lower()
    patterns = [
        r"\bnew (dataset|corpus|benchmark)\b",
        r"\bwe (introduce|release|propose|present|build|create|construct|compile|develop)\s+(a|an|the)?\s?(dataset|benchmark|corpus)\b",
        r"\bbenchmark for\b",
        r"\bwe (collect|curate|annotate|gather)\b.*\b(dataset|corpus)\b",
        r"\bwe make (a )?(dataset|corpus|benchmark) (public|available)\b",
    ]
    return any(re.search(p, s) for p in patterns)


def infer_n_classes(text: str) -> str:
    s = str(text).lower()
    if re.search(r"\bmulti[-\s]?label\b", s):
        return "multi-label"
    if re.search(r"\bbinary classification\b", s) or re.search(r"\btwo[-\s]?class\b", s) or re.search(r"\byes[-\s]?no\b", s) or "positive/negative" in s:
        return "2"
    if re.search(r"\bmulti[-\s]?class\b", s) or re.search(r"\b(\d+)[-\s]?(class|way)\b", s):
        return "3+"
    if re.search(r"\b(three|3|four|4|five|5|six|6|seven|7|eight|8|nine|9|ten|10)[-\s]?(class|way)\b", s):
        return "3+"
    return "unknown"


def infer_metric_type(text: str) -> str:
    s = str(text).lower()
    mets = []
    if "f1" in s or "f-score" in s: mets.append("F1")
    if "accuracy" in s: mets.append("Accuracy")
    if "precision" in s: mets.append("Precision")
    if "recall" in s: mets.append("Recall")
    if "auc" in s or "roc" in s: mets.append("AUC")
    if "macro" in s: mets.append("Macro")
    if "micro" in s: mets.append("Micro")
    return ", ".join(sorted(set(mets)))


def nc_bucket(x) -> str:
    x = str(x).lower() if isinstance(x, str) else str(x)
    if x in ["2", "binary", "binary (2)"]:
        return "2 (binary)"
    if "multi-label" in x:
        return "multi-label"
    if "3" in x or "+" in x or "multi-class" in x:
        return "3+ (multi-class)"
    return "unknown"


def as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def tokenize_metrics(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="all_classification_refined.csv", help="Path to the CSV (e.g., all_papers_with_classes_and_metrics.csv)")
    ap.add_argument("--outdir", default="cartography_outputs", help="Directory to save figures and pivots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)

    # Ensure columns (infer if missing)
    for c in ["year","venue","citations","type","domain","is_benchmark",
              "num_proposed_benchmarks","n_classes","metric_type","title","abstract"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Compose text field used by heuristics
    df["__text__"] = (df.get("title", "").astype(str) + " " + df.get("abstract", "").astype(str)).str.strip()

    # Inference fallbacks
    if df["type"].isna().all():
        df["type"] = df["__text__"].apply(infer_type)
    if df["domain"].isna().all():
        df["domain"] = df["__text__"].apply(infer_domain)
    if df["is_benchmark"].isna().all():
        df["is_benchmark"] = df["__text__"].apply(detect_benchmark_text)
    if df["num_proposed_benchmarks"].isna().all():
        df["num_proposed_benchmarks"] = df["is_benchmark"].astype(bool).astype(int)
    if df["n_classes"].isna().all():
        df["n_classes"] = df["__text__"].apply(infer_n_classes)
    if df["metric_type"].isna().all():
        df["metric_type"] = df.get("abstract","").astype(str).apply(infer_metric_type)

    # Coerce year
    df["year"] = df["year"].apply(as_int)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # 1) Combined: Papers per year and benchmarks per year
    bench = df[df["is_benchmark"]==True].copy()
    fig1 = plt.figure(figsize=(10,6))
    year_counts = df.groupby("year").size().reset_index(name="count")
    bench_year_counts = bench.groupby("year").size().reset_index(name="count")
    plt.plot(year_counts["year"], year_counts["count"], marker="o", label="Total Papers", linewidth=2, markersize=8)
    plt.plot(bench_year_counts["year"], bench_year_counts["count"], label="Benchmark Papers", linewidth=2)
    plt.title("Papers per Year")
    plt.xlabel("Year"); plt.ylabel("Number of Papers"); plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="best")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: str(int(x))))
    plt.tight_layout()
    fig1_path = os.path.join(args.outdir, "fig1_papers_per_year.png")
    plt.savefig(fig1_path, dpi=150); plt.close()

    # 2) Benchmarks per year by venue
    fig3 = plt.figure(figsize=(14,6))
    by_yv = bench.groupby(["year","venue"]).size().reset_index(name="count")
    # Filter to only venues with at least 3 total papers across all years
    venue_totals = by_yv.groupby("venue")["count"].sum()
    significant_venues = venue_totals[venue_totals >= 3].index
    by_yv_filtered = by_yv[by_yv["venue"].isin(significant_venues)]
    for venue, sub in by_yv_filtered.groupby("venue"):
        sub = sub.sort_values("year")
        plt.plot(sub["year"], sub["count"], marker="o", label=str(venue))
    plt.title("New Benchmarks per Year by Venue")
    plt.xlabel("Year"); plt.ylabel("Count of Benchmark Papers"); plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="upper left", fontsize=5, ncol=3, columnspacing=0.8, handlelength=1.5, frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(args.outdir, "fig3_benchmarks_per_year_by_venue.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight"); plt.close()

    # 4) Line chart: Papers by type over time (benchmark and total)
    # Filter out "unknown" type
    df_type_filtered = df[df["type"] != "unknown"].copy()
    
    # Convert is_benchmark to boolean for filtering
    is_benchmark_bool = df_type_filtered["is_benchmark"].astype(str).str.lower().isin(["true", "1", "yes"])
    
    # Get benchmark papers by year and type
    bench_type = df_type_filtered[is_benchmark_bool].groupby(["year", "type"]).size().reset_index(name="count")
    
    # Get total papers by year and type
    total_type = df_type_filtered.groupby(["year", "type"]).size().reset_index(name="count")
    
    fig4 = plt.figure(figsize=(12,6))
    
    # Get unique types (excluding unknown)
    unique_types = sorted([t for t in df_type_filtered["type"].unique() if t != "unknown"])
    
    # Plot benchmark papers by type
    for type_val in unique_types:
        type_data = bench_type[bench_type["type"] == type_val].sort_values("year")
        if len(type_data) > 0:
            plt.plot(type_data["year"], type_data["count"], marker="o", label=f"Benchmark: {type_val}", linewidth=2, markersize=6)
    
    # Plot total papers by type
    for type_val in unique_types:
        type_data = total_type[total_type["type"] == type_val].sort_values("year")
        if len(type_data) > 0:
            plt.plot(type_data["year"], type_data["count"], marker="s", linestyle="--", label=f"Total: {type_val}", linewidth=2, markersize=6)
    
    plt.title("Papers by Type Over Time (Benchmark vs Total)")
    plt.xlabel("Year"); plt.ylabel("Number of Papers")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig4_path = os.path.join(args.outdir, "fig4_papers_by_type_over_time.png")
    plt.savefig(fig4_path, dpi=150); plt.close()

    # 5) Heatmap: domain x year (benchmarks only)
    dom_year = bench.pivot_table(index="domain", columns="year", values="title", aggfunc="count", fill_value=0)
    dom_year_path = os.path.join(args.outdir, "pivot_domain_by_year_benchmarks.csv")
    dom_year.to_csv(dom_year_path)

    fig5 = plt.figure(figsize=(12,7))
    ax = plt.gca()
    im = ax.imshow(dom_year.values, aspect="auto")
    ax.set_yticks(range(len(dom_year.index))); ax.set_yticklabels(dom_year.index)
    ax.set_xticks(range(len(dom_year.columns))); ax.set_xticklabels(dom_year.columns, rotation=45, ha="right")
    plt.title("Benchmark Papers: Domain × Year (Counts)"); plt.xlabel("Year"); plt.ylabel("Domain")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig5_path = os.path.join(args.outdir, "fig5_heatmap_domain_year_benchmarks.png")
    plt.savefig(fig5_path, dpi=150); plt.close()

    # 6) Metrics frequency (stacked: benchmark vs non-benchmark)
    # Ensure is_benchmark_bool is defined
    if "is_benchmark_bool" not in df.columns:
        df["is_benchmark_bool"] = df["is_benchmark"].astype(str).str.lower().isin(["true", "1", "yes"])
    
    # Expand metrics to individual rows
    metric_rows = []
    for idx, row in df.iterrows():
        metrics = tokenize_metrics(row["metric_type"])
        for metric in metrics:
            if metric.lower() != "unknown":
                metric_rows.append({
                    "metric": metric,
                    "is_benchmark": row["is_benchmark_bool"]
                })
    
    metrics_expanded = pd.DataFrame(metric_rows)
    
    # Count by metric and benchmark status
    metrics_count = metrics_expanded.groupby(["metric", "is_benchmark"]).size().reset_index(name="count")
    
    # Get total counts per metric to find top 10
    metric_totals = metrics_expanded.groupby("metric").size().sort_values(ascending=False)
    top_10_metrics = metric_totals.head(10).index
    
    # Filter to top 10 metrics
    metrics_count_top10 = metrics_count[metrics_count["metric"].isin(top_10_metrics)]
    
    # Pivot to get benchmark and non-benchmark counts
    metrics_pivot = metrics_count_top10.pivot(index="metric", columns="is_benchmark", values="count").fillna(0)
    # Rename columns: True -> Benchmark, False -> Non-Benchmark
    column_mapping = {}
    if True in metrics_pivot.columns:
        column_mapping[True] = "Benchmark"
    if False in metrics_pivot.columns:
        column_mapping[False] = "Non-Benchmark"
    metrics_pivot = metrics_pivot.rename(columns=column_mapping)
    # Ensure both columns exist
    if "Benchmark" not in metrics_pivot.columns:
        metrics_pivot["Benchmark"] = 0
    if "Non-Benchmark" not in metrics_pivot.columns:
        metrics_pivot["Non-Benchmark"] = 0
    metrics_pivot = metrics_pivot[["Benchmark", "Non-Benchmark"]]  # Reorder
    
    # Sort by total count (descending)
    metrics_pivot["total"] = metrics_pivot.sum(axis=1)
    metrics_pivot = metrics_pivot.sort_values("total", ascending=False).drop("total", axis=1)

    fig6 = plt.figure(figsize=(12,6))
    x_pos = range(len(metrics_pivot))
    plt.bar(x_pos, metrics_pivot["Benchmark"].values, label="Benchmark", color="#2E86AB")
    plt.bar(x_pos, metrics_pivot["Non-Benchmark"].values, bottom=metrics_pivot["Benchmark"].values, label="Non-Benchmark", color="#A23B72")
    plt.xticks(x_pos, metrics_pivot.index, rotation=45, ha="right")
    plt.title("Evaluation Metrics Mentioned in Abstracts (Top 10)")
    plt.xlabel("Metric"); plt.ylabel("Count")
    plt.legend(title="Paper Type", loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5, axis="y")
    plt.tight_layout()
    fig6_path = os.path.join(args.outdir, "fig6_metric_frequency.png")
    plt.savefig(fig6_path, dpi=150); plt.close()

    # 7) Mean num_proposed_benchmarks per venue
    # Ensure is_benchmark_bool is defined
    if "is_benchmark_bool" not in df.columns:
        df["is_benchmark_bool"] = df["is_benchmark"].astype(str).str.lower().isin(["true", "1", "yes"])
    # Convert num_proposed_benchmarks to numeric, handling non-numeric values
    df["num_proposed_benchmarks_num"] = pd.to_numeric(df["num_proposed_benchmarks"], errors="coerce").fillna(0)
    # Only consider papers where is_benchmark is True and num_proposed_benchmarks > 0
    bench_with_count = df[(df["is_benchmark_bool"]) & (df["num_proposed_benchmarks_num"] > 0)]
    
    # 12) Average best performance by year (benchmark papers only)
    if "performance_best" in df.columns:
        # Convert performance_best to numeric, handling non-numeric values
        df["performance_best_num"] = pd.to_numeric(df["performance_best"], errors="coerce")
        # Filter for benchmark papers only
        perf_data = df[(df["is_benchmark_bool"]) & (df["performance_best_num"].notna())]
        
        if len(perf_data) > 0:
            mean_perf_by_year = perf_data.groupby("year")["performance_best_num"].mean()
            fig12 = plt.figure(figsize=(10,5))
            plt.plot(mean_perf_by_year.index, mean_perf_by_year.values, marker="o", linewidth=2, markersize=8)
            plt.title("Average Best Performance by Year (Benchmark Papers Only)")
            plt.xlabel("Year"); plt.ylabel("Mean performance_best")
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: str(int(x))))
            plt.tight_layout()
            fig12_path = os.path.join(args.outdir, "fig12_avg_performance_by_year.png")
            plt.savefig(fig12_path, dpi=150); plt.close()
        else:
            fig12_path = None
    else:
        fig12_path = None

    # 13) Pie chart: % of 'type' values from total papers
    type_counts = df["type"].value_counts()
    # Exclude "unspecified" and "unknown" types if they exist
    type_counts = type_counts[~type_counts.index.isin(["unspecified", "unknown"])]
    
    fig13 = plt.figure(figsize=(10,8))
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title("Distribution of Classification Types")
    plt.axis('equal')
    plt.tight_layout()
    fig13_path = os.path.join(args.outdir, "fig13_type_distribution_pie.png")
    plt.savefig(fig13_path, dpi=150); plt.close()

    # 14) Stacked bar chart: % of domain (benchmark vs non-benchmark)
    # Ensure is_benchmark_bool is defined
    if "is_benchmark_bool" not in df.columns:
        df["is_benchmark_bool"] = df["is_benchmark"].astype(str).str.lower().isin(["true", "1", "yes"])
    
    domain_total = df.groupby("domain").size()
    domain_benchmark = df[df["is_benchmark_bool"]].groupby("domain").size()
    domain_nonbenchmark = domain_total - domain_benchmark.reindex(domain_total.index, fill_value=0)
    
    # Sort by total count
    domain_total_sorted = domain_total.sort_values(ascending=False)
    domain_benchmark_sorted = domain_benchmark.reindex(domain_total_sorted.index, fill_value=0)
    domain_nonbenchmark_sorted = domain_nonbenchmark.reindex(domain_total_sorted.index, fill_value=0)
    
    fig14 = plt.figure(figsize=(10,8))
    y_pos = range(len(domain_total_sorted))
    plt.barh(y_pos, domain_benchmark_sorted.values, label="Benchmark", color="#2E86AB")
    plt.barh(y_pos, domain_nonbenchmark_sorted.values, left=domain_benchmark_sorted.values, label="Non-Benchmark", color="#A23B72")
    plt.yticks(y_pos, domain_total_sorted.index)
    plt.title("Distribution of Domains (Benchmark vs Non-Benchmark)")
    plt.xlabel("Number of Papers"); plt.ylabel("Domain")
    plt.legend(title="Paper Type", loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5, axis="x")
    # Double the number of x-axis labels
    plt.locator_params(axis='x', nbins=20)
    plt.tight_layout()
    fig14_path = os.path.join(args.outdir, "fig14_domain_distribution_bar.png")
    plt.savefig(fig14_path, dpi=150); plt.close()

    # 15) Stacked bar chart: % of task (benchmark vs non-benchmark)
    task_total = df.groupby("task").size()
    task_benchmark = df[df["is_benchmark_bool"]].groupby("task").size()
    task_nonbenchmark = task_total - task_benchmark.reindex(task_total.index, fill_value=0)
    
    # Sort by total count
    task_total_sorted = task_total.sort_values(ascending=False)
    task_benchmark_sorted = task_benchmark.reindex(task_total_sorted.index, fill_value=0)
    task_nonbenchmark_sorted = task_nonbenchmark.reindex(task_total_sorted.index, fill_value=0)
    
    fig15 = plt.figure(figsize=(10,8))
    y_pos = range(len(task_total_sorted))
    plt.barh(y_pos, task_benchmark_sorted.values, label="Benchmark", color="#2E86AB")
    plt.barh(y_pos, task_nonbenchmark_sorted.values, left=task_benchmark_sorted.values, label="Non-Benchmark", color="#A23B72")
    plt.yticks(y_pos, task_total_sorted.index)
    plt.title("Distribution of Tasks (Benchmark vs Non-Benchmark)")
    plt.xlabel("Number of Papers"); plt.ylabel("Task")
    plt.legend(title="Paper Type", loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5, axis="x")
    plt.tight_layout()
    fig15_path = os.path.join(args.outdir, "fig15_task_distribution_bar.png")
    plt.savefig(fig15_path, dpi=150); plt.close()

    # Text insight: Top benchmark-producing venues
    venue_benchmark_counts = bench.groupby("venue").size().sort_values(ascending=False)
    top_venues = venue_benchmark_counts.head(10)
    
    # Calculate benchmark counts by year for top venues
    insights_text = []
    insights_text.append("\n" + "="*60)
    insights_text.append("BENCHMARK PREVALENCE AND DIVERSITY INSIGHTS")
    insights_text.append("="*60)
    insights_text.append(f"\nTop 10 Benchmark-Producing Venues (all time):")
    for i, (venue, count) in enumerate(top_venues.items(), 1):
        insights_text.append(f"  {i:2d}. {venue}: {count} benchmark papers")
    
    # Identify venues leading in recent years (2020-2025)
    recent_years = bench[bench["year"] >= 2020]
    if len(recent_years) > 0:
        recent_venue_counts = recent_years.groupby("venue").size().sort_values(ascending=False)
        top_recent_venues = recent_venue_counts.head(5)
        insights_text.append(f"\nTop 5 Benchmark-Producing Venues (2020-2025):")
        for i, (venue, count) in enumerate(top_recent_venues.items(), 1):
            insights_text.append(f"  {i}. {venue}: {count} benchmark papers")
        
        # Identify the leading venue by year (2020-2025)
        insights_text.append(f"\nLeading Venue by Year (2020-2025):")
        for year in sorted(recent_years["year"].unique()):
            year_bench = recent_years[recent_years["year"] == year]
            venue_counts_year = year_bench.groupby("venue").size()
            if len(venue_counts_year) > 0:
                leading_venue = venue_counts_year.idxmax()
                leading_count = venue_counts_year.max()
                insights_text.append(f"  {year}: {leading_venue} ({leading_count} benchmarks)")
    
    # Write insights to file
    insights_path = os.path.join(args.outdir, "benchmark_insights.txt")
    with open(insights_path, "w") as f:
        f.write("\n".join(insights_text))
    
    # Print insights to console
    print("\n" + "\n".join(insights_text))

    # Print paths
    print("\n" + "="*60)
    print("Saved:")
    saved_paths = [fig1_path, fig3_path, fig4_path, fig5_path, fig6_path, dom_year_path]
    if fig12_path:
        saved_paths.append(fig12_path)
    saved_paths.append(fig13_path)
    saved_paths.append(fig14_path)
    saved_paths.append(fig15_path)
    saved_paths.append(insights_path)
    for p in saved_paths:
        print("  -", p)


if __name__ == "__main__":
    main()
