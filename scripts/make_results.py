"""Build final_results.{csv,md,xlsx} from raw metrics.txt + training log eval lines.
Usage: python scripts/make_results.py --metrics artifacts/metrics/metrics.txt
       --outdir artifacts/results [--reference docs/reference_metrics.json]
No fabrication: values come only from the given metrics file; reference column
stays 'N/A (no benchmark table in env)' unless a reference JSON is supplied.
"""
import argparse, csv, re
from pathlib import Path

KEYS = ["AUC", "AP", "PixAUC", "PixAP", "BestDice", "BestThresh",
        "normal_score", "abnormal_score"]

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
        import json
        ref = json.loads(Path(args.reference).read_text())
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rows = [["Metric", "Published/Reference", "Reproduced", "Absolute_Difference"]]
    for k in KEYS:
        rep = m.get(k, "N/A")
        r = ref.get(k, "N/A (no benchmark table in env)")
        try:
            d = abs(float(rep) - float(r))
            d = f"{d:.5f}"
        except (TypeError, ValueError):
            d = "N/A"
        rows.append([k, r, rep if isinstance(rep, str) else f"{rep:.5f}", d])
    with open(out / "final_results.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    lines = ["# DAE BraTS2021 final results",
             f"Method={args.method} Dataset={args.dataset} Input=128 Seed=0 Fold=0",
             "",
             "| Metric | Published/Reference | Reproduced | Absolute_Difference |",
             "|---|---|---|---|"]
    for r in rows[1:]:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    (out / "final_results.md").write_text("\n".join(lines) + "\n")
    try:
        import openpyxl
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = "results"
        for r in rows:
            ws.append(r)
        wb.save(out / "final_results.xlsx")
        print("xlsx written")
    except ImportError:
        print("openpyxl missing, xlsx skipped")
    print(f"metrics used: {m}")
    print(f"wrote {out}/final_results.{{csv,md}}")

if __name__ == "__main__":
    main()
