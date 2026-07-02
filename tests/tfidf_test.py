# tfidf_test.py
from src.stage2_specialty.tfidf_baseline import run_tfidf_baseline

# Run Stage 1 baseline (document type)
print("\n" + "="*60)
print("STAGE 1 — Document Type Classifier")
print("="*60)
s1_results = run_tfidf_baseline(stage=1)

# Run Stage 2 baseline (specialty)
print("\n" + "="*60)
print("STAGE 2 — Specialty Classifier")
print("="*60)
s2_results = run_tfidf_baseline(stage=2)

print("\n\n=== BASELINE SUMMARY ===")
print(f"Stage 1 | Val Macro F1: {s1_results['val_macro_f1']:.4f} | Test Macro F1: {s1_results['test_macro_f1']:.4f}")
print(f"Stage 2 | Val Macro F1: {s2_results['val_macro_f1']:.4f} | Test Macro F1: {s2_results['test_macro_f1']:.4f}")
print("\nClinicalBERT needs to beat these numbers to justify the added complexity.")