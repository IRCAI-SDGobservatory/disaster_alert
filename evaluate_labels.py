"""
evaluate_labels.py
------------------
Evaluates how well the ML zone labels (LOW / MIDDLE / HIGH) in the "full"
file match the human expert annotations (y / n) in the "blind" reference file.

Two mappings are compared:
  Conservative : only HIGH = relevant (y),  LOW + MIDDLE = not relevant (n)
  Liberal      : HIGH + MIDDLE = relevant (y),  LOW = not relevant (n)

Metrics reported for each mapping:
  Accuracy, Precision, Recall, F1-score, AUC-ROC, Cohen's Kappa,
  False Positive Rate, False Negative Rate, full confusion matrix.

Usage:
    python evaluate_labels.py <full_file.csv> <blind_file.csv> [--output report.csv]

Arguments:
    full_file   CSV with ML predictions (filename WITHOUT "blind").
    blind_file  CSV with expert labels   (the REFERENCE / "blind" file).

Options:
    --output    Optional path to save a per-tweet CSV report.

Example:
    python evaluate_labels.py BelgiumJuly2021_features_annotate_full.csv \\
                              BelgiumJuly2021_features_annotate_blind.csv \\
                              --output report.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        cohen_kappa_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        accuracy_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

ZONE_ORDER = ["LOW", "MIDDLE", "HIGH"]


def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: File not found: {path}")
    if p.stat().st_size == 0:
        sys.exit(f"ERROR: File is empty: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", lineterminator="\n")


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def kappa_label(k: float) -> str:
    if k >= 0.81: return "Almost perfect agreement"
    if k >= 0.61: return "Substantial agreement"
    if k >= 0.41: return "Moderate agreement"
    if k >= 0.21: return "Fair agreement"
    if k >= 0.01: return "Slight agreement"
    return "Poor / no agreement"


def print_confusion(cm: np.ndarray, labels: list, title: str) -> None:
    print(f"\n  {title}")
    header = f"  {'Ref / Pred':>18s}" + "".join(f"  {l:>14s}" for l in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, rl in enumerate(labels):
        row = f"  {rl:>18s}" + "".join(f"  {cm[i,j]:>14d}" for j in range(len(labels)))
        print(row)


def save_confusion_heatmap(cm: np.ndarray, labels: list,
                            title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(max(4, len(labels)*2), max(3.5, len(labels)*1.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted label", ylabel="Reference (expert) label",
        title=title,
    )
    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap : {path}")


def evaluate_zone_mapping(y_ref: list, y_pred: list,
                           mapping_name: str, n: int,
                           merged: pd.DataFrame,
                           output_path, part_num: int) -> dict:
    """
    Run all metrics for one zone-to-binary mapping.
    Returns a dict of key metrics for the summary table.
    """
    section(f"PART {part_num} — Zone mapping: {mapping_name}")
    print(f"  {'HIGH only = relevant (y),  LOW + MIDDLE = not relevant (n)' if 'Conservative' in mapping_name else 'HIGH + MIDDLE = relevant (y),  LOW = not relevant (n)'}")

    acc = accuracy_score(y_ref, y_pred) if HAS_SKLEARN else \
          sum(p == r for p, r in zip(y_pred, y_ref)) / n
    print(f"\n  Accuracy    : {acc:.4f}  ({acc*100:.1f}%)")

    results = {"mapping": mapping_name, "accuracy": acc}

    if HAS_SKLEARN:
        prec  = precision_score(y_ref, y_pred, zero_division=0)
        rec   = recall_score(y_ref, y_pred, zero_division=0)
        f1    = f1_score(y_ref, y_pred, zero_division=0)
        kappa = cohen_kappa_score(y_ref, y_pred)
        try:
            auc = roc_auc_score(y_ref, y_pred)
        except Exception:
            auc = float("nan")

        print(f"  Precision   : {prec:.4f}  (of tweets flagged relevant, how many truly are)")
        print(f"  Recall      : {rec:.4f}  (of truly relevant tweets, how many were caught)")
        print(f"  F1-score    : {f1:.4f}")
        print(f"  AUC-ROC     : {auc:.4f}")
        print(f"  Cohen's κ   : {kappa:.4f}  ({kappa_label(kappa)})")

        cm = confusion_matrix(y_ref, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

        print(f"\n  TP (expert=y, ML=relevant — correct)      : {tp}")
        print(f"  TN (expert=n, ML=not relevant — correct)  : {tn}")
        print(f"  FP (expert=n, ML=relevant — over-predicts): {fp}")
        print(f"  FN (expert=y, ML=not relevant — misses)   : {fn}")
        print(f"  False Positive Rate : {fpr:.4f}")
        print(f"  False Negative Rate : {fnr:.4f}")

        print(f"\n  Per-class classification report:")
        print(classification_report(y_ref, y_pred, labels=[0, 1],
                                    target_names=["Not relevant (n)", "Relevant (y)"],
                                    zero_division=0, digits=4))

        print_confusion(cm, ["Not relevant (0)", "Relevant (1)"],
                        "Confusion matrix (rows=expert reference, cols=ML zone prediction)")

        if HAS_MPL and output_path:
            suffix = ".conservative_cm.png" if "Conservative" in mapping_name else ".liberal_cm.png"
            png = str(Path(output_path).with_suffix(suffix))
            save_confusion_heatmap(cm, ["Not relevant", "Relevant"],
                                   f"Zone ({mapping_name}) vs Expert", png)

        results.update({
            "precision": prec, "recall": rec, "f1": f1,
            "auc": auc, "kappa": kappa, "fpr": fpr, "fnr": fnr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(full_path: str, blind_path: str, output_path) -> None:

    # ── Load files ────────────────────────────────────────────────────────────
    print(f"\nPredicted labels file : {full_path}")
    print(f"Reference (blind) file: {blind_path}")

    df_full  = read_csv(full_path)
    df_blind = read_csv(blind_path)

    # ── Validate columns ──────────────────────────────────────────────────────
    required_full  = {"tweet_id", "zone", "composite_score"}
    required_blind = {"tweet_id", "expert_label"}
    missing_full  = required_full  - set(df_full.columns)
    missing_blind = required_blind - set(df_blind.columns)
    if missing_full:
        sys.exit(f"ERROR: Full file missing columns: {missing_full}")
    if missing_blind:
        sys.exit(f"ERROR: Blind file missing columns: {missing_blind}")

    # ── Merge on tweet_id ─────────────────────────────────────────────────────
    keep_full = ["tweet_id", "zone", "composite_score"]
    if "text" in df_full.columns:
        keep_full.append("text")
    merged = pd.merge(
        df_full[keep_full],
        df_blind[["tweet_id", "expert_label"]],
        on="tweet_id", how="inner",
    )

    section("Dataset overview")
    print(f"  Rows in full file             : {len(df_full)}")
    print(f"  Rows in reference (blind) file: {len(df_blind)}")
    print(f"  Matched rows (on tweet_id)    : {len(merged)}")
    if len(merged) == 0:
        sys.exit(
            "\nERROR: No rows matched between the two files.\n"
            "This usually means you passed mismatched files (e.g. Indonesia full vs India blind).\n"
            f"  Full file  : {full_path}\n"
            f"  Blind file : {blind_path}\n"
            "Please check that both files are from the same country/event."
        )
    if len(merged) < len(df_blind):
        print(f"  WARNING: {len(df_blind)-len(merged)} reference rows had no match.")

    # ── Prepare labels ────────────────────────────────────────────────────────
    merged["expert_label"] = merged["expert_label"].astype(str).str.strip().str.lower()
    invalid = merged[~merged["expert_label"].isin(["y", "n"])]
    if not invalid.empty:
        print(f"  WARNING: {len(invalid)} rows have unexpected expert_label values; excluded.")
        merged = merged[merged["expert_label"].isin(["y", "n"])].copy()

    merged["expert_bin"] = (merged["expert_label"] == "y").astype(int)
    merged["zone"]       = merged["zone"].astype(str).str.strip().str.upper()

    n = len(merged)

    # ── Label distributions ───────────────────────────────────────────────────
    section("Label distributions")
    print(f"\n  Expert labels (ground truth)  —  y (relevant): {(merged['expert_label']=='y').sum()}  |  n (not relevant): {(merged['expert_label']=='n').sum()}")
    print(f"\n  ML zone distribution:")
    for z in ZONE_ORDER:
        cnt = (merged["zone"] == z).sum()
        print(f"    {z:>8s} : {cnt:3d}  {'█' * cnt}")

    print(f"\n  Zone breakdown by expert label:")
    ct = pd.crosstab(merged["zone"], merged["expert_label"],
                     margins=True, margins_name="Total")
    ct.index.name = "Zone \\ Expert"
    print(ct.to_string())

    # ── Two zone mappings ─────────────────────────────────────────────────────
    y_ref = merged["expert_bin"].tolist()

    # Conservative: HIGH = relevant, LOW + MIDDLE = not relevant
    merged["zone_conservative"] = merged["zone"].isin(["HIGH"]).astype(int)
    y_conservative = merged["zone_conservative"].tolist()

    # Liberal: HIGH + MIDDLE = relevant, LOW = not relevant
    merged["zone_liberal"] = merged["zone"].isin(["HIGH", "MIDDLE"]).astype(int)
    y_liberal = merged["zone_liberal"].tolist()

    r_conservative = evaluate_zone_mapping(
        y_ref, y_conservative, "Conservative (HIGH only)", n, merged, output_path, part_num=1
    )
    r_liberal = evaluate_zone_mapping(
        y_ref, y_liberal, "Liberal (HIGH + MIDDLE)", n, merged, output_path, part_num=2
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Composite score distribution by expert label
    # ═══════════════════════════════════════════════════════════════════════════
    section("PART 3 — Composite score distribution by expert label")
    for lbl, grp in merged.groupby("expert_label"):
        lbl_name = "Relevant (y)" if lbl == "y" else "Not relevant (n)"
        s = grp["composite_score"]
        print(f"\n  {lbl_name}  (n={len(grp)})")
        print(f"    Mean   : {s.mean():.4f}   |  Median : {s.median():.4f}")
        print(f"    Std    : {s.std():.4f}   |  Min: {s.min():.4f}  Max: {s.max():.4f}")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 4))
        for lbl, grp in merged.groupby("expert_label"):
            label_str = "Expert: relevant (y)" if lbl == "y" else "Expert: not relevant (n)"
            ax.hist(grp["composite_score"], bins=15, alpha=0.6, label=label_str)
        ax.set(xlabel="Composite score", ylabel="Count",
               title="Composite score distribution by expert label")
        ax.legend()
        plt.tight_layout()
        png3 = str(Path(output_path).with_suffix(".score_dist.png")
                   if output_path else Path("score_distribution.png"))
        fig.savefig(png3, dpi=150)
        plt.close(fig)
        print(f"\n  Score distribution plot saved : {png3}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Disagreement analysis
    # ═══════════════════════════════════════════════════════════════════════════
    section("PART 4 — Disagreement analysis")

    for mapping, col in [("Conservative (HIGH only)", "zone_conservative"),
                          ("Liberal (HIGH + MIDDLE)",  "zone_liberal")]:
        merged[f"match_{col}"] = merged[col] == merged["expert_bin"]
        n_mismatch = (~merged[f"match_{col}"]).sum()
        print(f"\n  {mapping} — mismatches: {n_mismatch} / {n}")

        if n_mismatch > 0:
            bad = merged[~merged[f"match_{col}"]].copy()
            bad["error_type"] = bad.apply(
                lambda r, c=col: "FP (ML=relevant, expert=n)" if r[c] == 1
                                 else "FN (ML=not relevant, expert=y)", axis=1
            )
            fp_count = bad["error_type"].str.startswith("FP").sum()
            fn_count = bad["error_type"].str.startswith("FN").sum()
            print(f"    FP (over-predicts relevance): {fp_count}")
            print(f"    FN (misses relevant tweets) : {fn_count}")
            cols = ["tweet_id", "expert_label", "zone", "composite_score", "error_type"]
            if "text" in bad.columns:
                bad["text_snippet"] = bad["text"].astype(str).str[:80]
                cols.append("text_snippet")
            print(bad[cols].to_string(index=False))

    # ── Per-tweet output CSV ──────────────────────────────────────────────────
    if output_path:
        out_cols = ["tweet_id", "expert_label", "expert_bin",
                    "zone", "zone_conservative", "zone_liberal",
                    "composite_score",
                    "match_zone_conservative", "match_zone_liberal"]
        if "text" in merged.columns:
            out_cols.append("text")
        merged[out_cols].to_csv(output_path, index=False)
        print(f"\n  Per-tweet report saved : {output_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════════════════
    section("Summary")
    print(f"\n  {'Mapping':<30s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} "
          f"{'F1':>10s} {'AUC':>10s} {'Kappa':>10s}")
    print("  " + "-" * 82)
    for r in [r_conservative, r_liberal]:
        print(f"  {r['mapping']:<30s} "
              f"{r.get('accuracy', float('nan')):>10.4f} "
              f"{r.get('precision', float('nan')):>10.4f} "
              f"{r.get('recall', float('nan')):>10.4f} "
              f"{r.get('f1', float('nan')):>10.4f} "
              f"{r.get('auc', float('nan')):>10.4f} "
              f"{r.get('kappa', float('nan')):>10.4f}")
    print()
    for r in [r_conservative, r_liberal]:
        print(f"  {r['mapping']} — {kappa_label(r.get('kappa', 0))}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ML zone labels against expert human annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("full_file",
                   help="CSV with ML predictions (file without 'blind' in name).")
    p.add_argument("blind_file",
                   help="CSV with expert reference labels (the 'blind' file).")
    p.add_argument("--output", default=None,
                   help="Optional path for a per-tweet CSV report (e.g. report.csv).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        full_path   = args.full_file,
        blind_path  = args.blind_file,
        output_path = args.output,
    )
