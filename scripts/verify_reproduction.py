#!/usr/bin/env python3
"""Verify reproduction completeness. Exit non-zero on failure."""
import json, sys, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails = []
def need(path, label):
    if not (ROOT / path).exists():
        fails.append(f"missing {label}: {path}")

def no_placeholders(path):
    t = (ROOT / path).read_text(errors="replace")
    for s in ["TBD", "TODO", "PLACEHOLDER", "pending (no benchmark"]:
        if s in t:
            # final_results may legitimately note pending reference values;
            # only fail on TBD/TODO/PLACEHOLDER
            if s.startswith("pending") and "final_results" in path:
                continue
            fails.append(f"placeholder '{s}' in {path}")

need("artifacts/dataset_manifest.json", "dataset manifest")
need("artifacts/reproduction_manifest.json", "reproduction manifest")
need("artifacts/model_summary.txt", "model summary")
need("artifacts/execution/environment.txt", "environment log")
need("artifacts/execution/pip_freeze.txt", "pip freeze")
need("artifacts/execution/torch_info.txt", "torch info")
need("artifacts/execution/final_command.txt", "final command")
need("artifacts/results/final_results.csv", "results csv")
need("artifacts/results/final_results.md", "results md")
need("artifacts/metrics/metrics.txt", "raw metrics")
need("docs/implementation_trace.md", "implementation trace")
need("docs/paper_vs_repository.md", "paper vs repo")
need("docs/experiment_scope.md", "scope")
need("docs/DAE_methodology.md", "methodology")
need("docs/reproduction_report.md", "report")
need("CLIENT_README.md", "client readme")
need("scripts/inspect_brats.py", "inspect script")
need("scripts/run_dae_reproduction.sh", "rerun entrypoint")

# checkpoint expected after full run
if not (ROOT / "artifacts/checkpoints/model.pt").exists():
    fails.append("missing checkpoint artifacts/checkpoints/model.pt (full run not yet pulled)")

# metrics numeric check
mp = ROOT / "artifacts/metrics/metrics.txt"
if mp.exists():
    for line in mp.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in ("AUC", "AP", "PixAUC", "PixAP", "BestDice"):
                try:
                    float(v)
                except ValueError:
                    fails.append(f"non-numeric metric {k}={v}")

# dataset paths
try:
    dm = json.loads((ROOT / "artifacts/dataset_manifest.json").read_text())
    assert dm["normal_training_count"] == 4211, "train count mismatch"
    assert dm["test_normal_count"] == 828
    assert dm["test_abnormal_count"] == 1948
except Exception as e:
    fails.append(f"dataset manifest check failed: {e}")

# no placeholders in docs (excluding pending-reference note in results)
for d in ["CLIENT_README.md", "docs/reproduction_report.md", "docs/DAE_methodology.md"]:
    if (ROOT / d).exists():
        no_placeholders(d)

if fails:
    print("VERIFY FAILED:")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("VERIFY OK")
