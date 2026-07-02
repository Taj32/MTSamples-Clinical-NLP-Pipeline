import pandas as pd
from src.preprocessing.medspacy_pipeline import build_medspacy_pipeline, preprocess_report

df = pd.read_csv("data/processed/mtsamples_clean.csv", index_col=0)
sample = df["transcription"].iloc[0]

nlp = build_medspacy_pipeline()
result = preprocess_report(sample, nlp)

print("=== SECTIONS FOUND ===")
for section, text in result["sections"].items():
    print(f"\n[{section.upper()}]\n{text[:200]}")

print("\n=== SENTENCES ===")
for sent in list(result["doc"].sents)[:5]:
    print(f"  - {sent.text[:100]}")