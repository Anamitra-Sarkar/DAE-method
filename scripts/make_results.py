"""Build final_results.{csv,md,xlsx} from raw metrics.txt + reference.json.
Usage: python scripts/make_results.py --metrics artifacts/metrics/metrics.txt
       --outdir artifacts/results [--reference artifacts/results/reference.json]
No fabrication: reproduced values come only from metrics.txt; published values
only from reference.json (hand-transcribed from the cited PDF table). Metrics
without a published counterpart stay N/A with reason.
Columns: Metric | Published (mean+-std) | Reproduced | Diff (rep-pub) |
         |Diff| | % diff vs published mean | Source
"""
import argparse
import csv
import json
from pathlib import Path

ROWS = [
    # (metric, paper_term, reproduced key)
    ("AUC", "AUC", "AUC"),
    ("AP", "AP (image)", "AP"),
    ("PixAUC", "pixel AUC (not reported)", "PixAUC"),
    ("PixAP", "APpix", "PixAP"),
    ("BestDice", "⌈Dice⌉", "BestDice"),
    ("BestThresh", "—", "BestThresh"),
    ("normal_score", "—", "normal_score"),
    ("abnormal_score", "—", "abnormal_score"),
]


def parse_metrics(p: Path):
    m = {}
    for line in p.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            try:
                m[k] = float(v)
            except ValueError:
                m[k] = v
    return m


def fmt(x, nd=5):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--method", default="DAE")
    ap.add_argument("--dataset", default="BraTS2021")
    args = ap.parse_args()
    m = parse_metrics(Path(args.metrics))
    ref = {}
    if args.reference and Path(args.reference).exists():
        ref = json.loads(Path(args.reference).read_text()).get("metrics", {})
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    header = ["Metric", "Paper_Term", "Published_mean_pm_std", "Reproduced",
              "Diff_rep_minus_pub", "AbsDiff", "PctDiff_vs_pub_mean", "Source"]
    rows = [header]
    for metric, paper_term, rep_key in ROWS:
        rep = m.get(rep_key, "N/A")
        r = ref.get(metric, {})
        mean, std = r.get("mean"), r.get("std")
        cit = r.get("citation", {})
        src = cit.get("table", "") + (f" p.{cit['page']}" if "page" in cit else "")
        if metric == "PixAUC":
            pub, diff, ad, pct = "N/A (excluded by paper, p.9)", "N/A", "N/A", "N/A"
            src = "medianomaly.pdf p.9 (pixel AUC not employed)"
        elif mean is None:
            pub = "N/A (no published counterpart)"
            diff = ad = pct = "N/A"
            src = "—"
        else:
            pub = f"{mean:.3f}+-{std:.3f}"
            try:
                d = float(rep) - mean
                diff, ad = fmt(d), fmt(abs(d))
                pct = fmt(abs(d) / mean * 100, 2) + "%"
            except (TypeError, ValueError):
                diff = ad = pct = "N/A"
        rows.append([metric, paper_term, pub,
                     rep if isinstance(rep, str) else fmt(rep),
                     diff, ad, pct, src])
    with open(out / "final_results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    lines = ["# DAE BraTS2021 final results",
             f"Method={args.method} Dataset={args.dataset} Input=128 Seed=0 Fold=0",
             "Reference: medianomaly.pdf Table 6 (p.11, image) + Table 7 (p.12, pixel); "
             "paper = mean+-std over 3 seeds, reproduction = seed 0.",
             "",
             "| Metric | Paper term | Published | Reproduced | Diff | |Diff| | % diff | Source |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    (out / "final_results.md").write_text("\n".join(lines) + "\n")
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "results"
        for r in rows:
            ws.append(r)
        wb.save(out / "final_results.xlsx")
        print("xlsx written")
    except ImportError:
        print("openpyxl missing, xlsx skipped")
    print(f"metrics used: {m}")
    print(f"wrote {out}/final_results.{{csv,md,xlsx}}")


if __name__ == "__main__":
    main()
