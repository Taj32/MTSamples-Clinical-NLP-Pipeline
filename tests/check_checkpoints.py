# check_checkpoints.py
import torch
from pathlib import Path

checkpoint_dir = Path("models")
checkpoints = sorted(checkpoint_dir.glob("*.pt"))

if not checkpoints:
    print("No checkpoints found in models/")
else:
    print(f"{'File':<60} {'Stage':>6} {'Val F1':>8} {'Epoch':>6}")
    print("-" * 82)
    for ckpt in checkpoints:
        try:
            data = torch.load(ckpt, map_location="cpu")
            stage = "S1" if "stage1" in ckpt.name else "S2"
            val_f1 = data.get("val_macro_f1", "N/A")
            epoch = data.get("epoch", "N/A")
            print(f"{ckpt.name:<60} {stage:>6} {val_f1:>8.4f} {epoch:>6}")
        except Exception as e:
            print(f"{ckpt.name:<60}  ERROR: {e}")