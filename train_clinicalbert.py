# # train_clinicalbert.py
# from src.stage2_specialty.clinicalbert_trainer import train_clinicalbert

# # Start with Stage 2 (specialty) — more interesting and faster
# # since it only uses 2,073 reports vs 3,476 for Stage 1
# print("Starting Stage 2 ClinicalBERT training...")
# print("Expected time on RTX 4070 Ti Super: ~15-25 minutes\n")

# results = train_clinicalbert(
#     stage=2,
#     epochs=4,
#     batch_size=8,
#     lr=2e-5,
# )

# print(f"\nFinal Results:")
# print(f"  Best Val Macro F1: {results['best_val_f1']:.4f}")
# print(f"  Test Macro F1:     {results['test_macro_f1']:.4f}")
# print(f"  Test Accuracy:     {results['test_accuracy']:.4f}")

# # train_clinicalbert.py
# from src.stage2_specialty.clinicalbert_trainer import train_clinicalbert

# print("Starting Stage 2 ClinicalBERT — run 2 (tuned hyperparameters)")

# results = train_clinicalbert(
#     stage=2,
#     epochs=6,        # was 4 — more epochs to converge
#     batch_size=16,   # was 8 — larger batches, more stable gradients
#     lr=3e-5,         # was 2e-5 — slightly higher, linear warmup will protect it
#     warmup_ratio=0.15,  # was 0.1 — longer warmup for larger lr
#     dropout=0.2,     # was 0.1 — more regularization for 6 epochs
# )

# print(f"\nRun 2 Results:")
# print(f"  Best Val Macro F1: {results['best_val_f1']:.4f}")
# print(f"  Test Macro F1:     {results['test_macro_f1']:.4f}")
# print(f"  Test Accuracy:     {results['test_accuracy']:.4f}")

# ===============================================================================================

# # train_clinicalbert.py
# from src.model.clinicalbert_trainer import train_clinicalbert

# print("Starting Stage 1 ClinicalBERT — document type classifier")
# print("5 classes, 3,476 training samples, expect faster convergence\n")

# results = train_clinicalbert(
#     # stage=1,
#     # epochs=4,
#     # batch_size=8,
#     # lr=2e-5,
#     # warmup_ratio=0.15,
#     # dropout=0.2,
#     # override_weights={
#     #     "specialty_report":   4.0,   # heavily boost the majority-but-hard class
#     #     "consultation":       2.0,
#     #     "discharge_summary":  3.0,
#     #     "procedure_note":     0.5,   # penalize over-predicting this
#     #     "progress_note":      2.5,
#     # }
#     stage=1,
#     epochs=5,
#     batch_size=8,
#     lr=2e-5,
#     warmup_ratio=0.15,
#     dropout=0.2,
#     override_weights=None,   # use plain inverse-frequency weights, same as run A
#     use_focal_loss=True,     # new flag
#     focal_gamma=2.0,
# )

# print(f"\nFocal Loss Stage 1 Run Results:")
# print(f"  Best Val Macro F1: {results['best_val_f1']:.4f}")
# print(f"  Test Macro F1:     {results['test_macro_f1']:.4f}")
# print(f"  Test Accuracy:     {results['test_accuracy']:.4f}")
# print(f"\nCompare to run A (cross-entropy, same weights): test F1 = 0.5102")

# =======================================================================================================
# train_clinicalbert.py
from src.model.clinicalbert_trainer import train_clinicalbert

print("Starting Stage 1 ClinicalBERT — structural features experiment")
print("Adding 7 structural features (section markers, tense ratio, doc length)\n")

results = train_clinicalbert(
    stage=1,
    epochs=5,
    batch_size=8,
    lr=2e-5,
    warmup_ratio=0.15,
    dropout=0.2,
    use_focal_loss=True,
    focal_gamma=2.0,
)

print(f"\nStructural Features Run Results:")
print(f"  Best Val Macro F1: {results['best_val_f1']:.4f}")
print(f"  Test Macro F1:     {results['test_macro_f1']:.4f}")
print(f"\nCompare to focal loss run (no structural features): test F1 = 0.5276")

# =======================================================================================================


# # train_clinicalbert.py
# from model.clinicalbert_trainer import train_clinicalbert

# print("Starting Stage 2 ClinicalBERT — run 3 (early stopping focus)")
# print("Key changes: lower lr (1e-5), back to batch=8, 4 epochs, higher dropout\n")

# results = train_clinicalbert(
#     stage=2,
#     epochs=4,
#     batch_size=8,
#     lr=2e-5,
# )

# print(f"\nRun 3 Results:")
# print(f"  Best Val Macro F1: {results['best_val_f1']:.4f}")
# print(f"  Test Macro F1:     {results['test_macro_f1']:.4f}")
# print(f"  Test Accuracy:     {results['test_accuracy']:.4f}")