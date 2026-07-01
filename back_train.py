# batch_train.py
# Run this and leave it — trains multiple Stage 1 configurations and saves a comparison report.
# Check results/batch_training_report.txt when you return.

import json
from pathlib import Path
from datetime import datetime
from src.model.clinicalbert_trainer import train_clinicalbert

Path("results").mkdir(exist_ok=True)

runs = [
    {
        "name": "s1_run_a_balanced_weights",
        "description": "Inverse-freq weights only, lr=2e-5, 5 epochs",
        "params": dict(stage=1, epochs=5, batch_size=8, lr=2e-5,
                       warmup_ratio=0.15, dropout=0.2,
                       override_weights=None),
    },
    {
        "name": "s1_run_b_gentle_boost",
        "description": "Gentle specialty_report boost (2.5x), lower lr",
        "params": dict(stage=1, epochs=5, batch_size=8, lr=1e-5,
                       warmup_ratio=0.2, dropout=0.2,
                       override_weights={
                           "specialty_report":   2.5,
                           "consultation":       1.8,
                           "discharge_summary":  2.5,
                           "procedure_note":     0.8,
                           "progress_note":      2.0,
                       }),
    },
    {
        "name": "s1_run_c_more_epochs",
        "description": "Balanced weights, more epochs, higher dropout",
        "params": dict(stage=1, epochs=6, batch_size=8, lr=2e-5,
                       warmup_ratio=0.15, dropout=0.3,
                       override_weights=None),
    },
    {
        "name": "s1_run_d_moderate_boost",
        "description": "Moderate specialty_report boost (3x), medium lr",
        "params": dict(stage=1, epochs=5, batch_size=8, lr=1.5e-5,
                       warmup_ratio=0.15, dropout=0.25,
                       override_weights={
                           "specialty_report":   3.0,
                           "consultation":       1.5,
                           "discharge_summary":  2.0,
                           "procedure_note":     0.7,
                           "progress_note":      1.8,
                       }),
    },
]

results = []
start_time = datetime.now()

print(f"Batch training started at {start_time.strftime('%H:%M:%S')}")
print(f"Running {len(runs)} configurations — estimated time: 3-4 hours\n")
print("="*60)

for i, run in enumerate(runs, 1):
    print(f"\n[{i}/{len(runs)}] {run['name']}")
    print(f"  {run['description']}")
    run_start = datetime.now()

    try:
        metrics = train_clinicalbert(**run["params"])
        duration = (datetime.now() - run_start).seconds // 60

        result = {
            "run":          run["name"],
            "description":  run["description"],
            "params":       {k: v for k, v in run["params"].items()
                             if k not in ("stage", "override_weights")},
            "override_weights": run["params"].get("override_weights"),
            "best_val_f1":  round(metrics["best_val_f1"], 4),
            "test_macro_f1": round(metrics["test_macro_f1"], 4),
            "test_accuracy": round(metrics["test_accuracy"], 4),
            "duration_min": duration,
            "status":       "success",
        }

    except Exception as e:
        result = {
            "run":    run["name"],
            "status": "failed",
            "error":  str(e),
        }
        print(f"  ERROR: {e}")

    results.append(result)

    # Save intermediate results after every run so you can check progress
    with open("results/batch_training_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Done — test macro-F1: {result.get('test_macro_f1', 'N/A')} "
          f"({result.get('duration_min', '?')} min)")

# ── Final report ───────────────────────────────────────────────────────────────
total_duration = (datetime.now() - start_time).seconds // 60

report_lines = [
    "=" * 60,
    "BATCH TRAINING REPORT — Stage 1 ClinicalBERT",
    f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Total time: {total_duration} minutes",
    "=" * 60,
    "",
    f"{'Run':<30} {'Val F1':>8} {'Test F1':>9} {'Accuracy':>10} {'Min':>6}",
    "-" * 68,
]

# Sort by test F1 descending
successful = [r for r in results if r["status"] == "success"]
successful.sort(key=lambda x: x["test_macro_f1"], reverse=True)

for r in successful:
    marker = " ← BEST" if r == successful[0] else ""
    report_lines.append(
        f"{r['run']:<30} {r['best_val_f1']:>8.4f} "
        f"{r['test_macro_f1']:>9.4f} {r['test_accuracy']:>10.4f} "
        f"{r['duration_min']:>6}{marker}"
    )

failed = [r for r in results if r["status"] == "failed"]
if failed:
    report_lines += ["", "FAILED RUNS:"]
    for r in failed:
        report_lines.append(f"  {r['run']}: {r['error']}")

report_lines += [
    "",
    "PREVIOUS BASELINES:",
    f"  TF-IDF Stage 1 baseline:       test macro-F1 = 0.4940",
    f"  ClinicalBERT Stage 1 run 1:    test macro-F1 = 0.4954",
    f"  ClinicalBERT Stage 1 run 2:    test macro-F1 = 0.1824 (over-boosted, ignore)",
    "",
    "DESCRIPTIONS:",
]
for r in results:
    report_lines.append(f"  {r['run']}: {r.get('description', '')}")
    if r.get("override_weights"):
        report_lines.append(f"    weights: {r['override_weights']}")

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open("results/batch_training_report.txt", "w") as f:
    f.write(report_text)

print(f"\nFull report saved to results/batch_training_report.txt")
print(f"JSON data saved to results/batch_training_report.json")# batch_train.py
# Run this and leave it — trains multiple Stage 1 configurations and saves a comparison report.
# Check results/batch_training_report.txt when you return.

import json
from pathlib import Path
from datetime import datetime
from src.model.clinicalbert_trainer import train_clinicalbert

Path("results").mkdir(exist_ok=True)

runs = [
    {
        "name": "s1_run_a_balanced_weights",
        "description": "Inverse-freq weights only, lr=2e-5, 5 epochs",
        "params": dict(stage=1, epochs=5, batch_size=8, lr=2e-5,
                       warmup_ratio=0.15, dropout=0.2,
                       override_weights=None),
    },
    {
        "name": "s1_run_b_gentle_boost",
        "description": "Gentle specialty_report boost (2.5x), lower lr",
        "params": dict(stage=1, epochs=5, batch_size=8, lr=1e-5,
                       warmup_ratio=0.2, dropout=0.2,
                       override_weights={
                           "specialty_report":   2.5,
                           "consultation":       1.8,
                           "discharge_summary":  2.5,
                           "procedure_note":     0.8,
                           "progress_note":      2.0,
                       }),
    },
    {
        "name": "s1_run_c_more_epochs",
        "description": "Balanced weights, more epochs, higher dropout",
        "params": dict(stage=1, epochs=6, batch_size=8, lr=2e-5,
                       warmup_ratio=0.15, dropout=0.3,
                       override_weights=None),
    },
    {
        "name": "s1_run_d_moderate_boost",
        "description": "Moderate specialty_report boost (3x), medium lr",
        "params": dict(stage=1, epochs=5, batch_size=8, lr=1.5e-5,
                       warmup_ratio=0.15, dropout=0.25,
                       override_weights={
                           "specialty_report":   3.0,
                           "consultation":       1.5,
                           "discharge_summary":  2.0,
                           "procedure_note":     0.7,
                           "progress_note":      1.8,
                       }),
    },
]

results = []
start_time = datetime.now()

print(f"Batch training started at {start_time.strftime('%H:%M:%S')}")
print(f"Running {len(runs)} configurations — estimated time: 3-4 hours\n")
print("="*60)

for i, run in enumerate(runs, 1):
    print(f"\n[{i}/{len(runs)}] {run['name']}")
    print(f"  {run['description']}")
    run_start = datetime.now()

    try:
        metrics = train_clinicalbert(**run["params"])
        duration = (datetime.now() - run_start).seconds // 60

        result = {
            "run":          run["name"],
            "description":  run["description"],
            "params":       {k: v for k, v in run["params"].items()
                             if k not in ("stage", "override_weights")},
            "override_weights": run["params"].get("override_weights"),
            "best_val_f1":  round(metrics["best_val_f1"], 4),
            "test_macro_f1": round(metrics["test_macro_f1"], 4),
            "test_accuracy": round(metrics["test_accuracy"], 4),
            "duration_min": duration,
            "status":       "success",
        }

    except Exception as e:
        result = {
            "run":    run["name"],
            "status": "failed",
            "error":  str(e),
        }
        print(f"  ERROR: {e}")

    results.append(result)

    # Save intermediate results after every run so you can check progress
    with open("results/batch_training_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Done — test macro-F1: {result.get('test_macro_f1', 'N/A')} "
          f"({result.get('duration_min', '?')} min)")

# ── Final report ───────────────────────────────────────────────────────────────
total_duration = (datetime.now() - start_time).seconds // 60

report_lines = [
    "=" * 60,
    "BATCH TRAINING REPORT — Stage 1 ClinicalBERT",
    f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Total time: {total_duration} minutes",
    "=" * 60,
    "",
    f"{'Run':<30} {'Val F1':>8} {'Test F1':>9} {'Accuracy':>10} {'Min':>6}",
    "-" * 68,
]

# Sort by test F1 descending
successful = [r for r in results if r["status"] == "success"]
successful.sort(key=lambda x: x["test_macro_f1"], reverse=True)

for r in successful:
    marker = " ← BEST" if r == successful[0] else ""
    report_lines.append(
        f"{r['run']:<30} {r['best_val_f1']:>8.4f} "
        f"{r['test_macro_f1']:>9.4f} {r['test_accuracy']:>10.4f} "
        f"{r['duration_min']:>6}{marker}"
    )

failed = [r for r in results if r["status"] == "failed"]
if failed:
    report_lines += ["", "FAILED RUNS:"]
    for r in failed:
        report_lines.append(f"  {r['run']}: {r['error']}")

report_lines += [
    "",
    "PREVIOUS BASELINES:",
    f"  TF-IDF Stage 1 baseline:       test macro-F1 = 0.4940",
    f"  ClinicalBERT Stage 1 run 1:    test macro-F1 = 0.4954",
    f"  ClinicalBERT Stage 1 run 2:    test macro-F1 = 0.1824 (over-boosted, ignore)",
    "",
    "DESCRIPTIONS:",
]
for r in results:
    report_lines.append(f"  {r['run']}: {r.get('description', '')}")
    if r.get("override_weights"):
        report_lines.append(f"    weights: {r['override_weights']}")

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open("results/batch_training_report.txt", "w") as f:
    f.write(report_text)

print(f"\nFull report saved to results/batch_training_report.txt")
print(f"JSON data saved to results/batch_training_report.json")