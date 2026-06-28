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

# train_clinicalbert.py
from src.stage2_specialty.clinicalbert_trainer import train_clinicalbert

print("Starting Stage 2 ClinicalBERT — run 3 (early stopping focus)")
print("Key changes: lower lr (1e-5), back to batch=8, 4 epochs, higher dropout\n")

results = train_clinicalbert(
    stage=2,
    epochs=4,
    batch_size=8,    # back to 8 — faster per epoch, less overfitting pressure
    lr=1e-5,         # lower than run 2 (3e-5 overfit by epoch 4)
    warmup_ratio=0.2,  # longer warmup for gentle start
    dropout=0.3,     # higher dropout — run 2 showed overfitting, need more regularization
)

print(f"\nRun 3 Results:")
print(f"  Best Val Macro F1: {results['best_val_f1']:.4f}")
print(f"  Test Macro F1:     {results['test_macro_f1']:.4f}")
print(f"  Test Accuracy:     {results['test_accuracy']:.4f}")