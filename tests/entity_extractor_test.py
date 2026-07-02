# pipeline_test.py  (replace previous content)
import pandas as pd
from src.preprocessing.medspacy_pipeline import build_medspacy_pipeline
from src.ner.entity_extractor import (
    load_ner_models,
    extract_entities,
    get_positive_entities,
    deduplicate_entities,
    extract_and_sanity_check,
)

df = pd.read_csv("data/processed/mtsamples_clean.csv", index_col=0)

# Load models (slow first time, cached after)
print("Loading models...")
nlp_sci, nlp_bc5 = load_ner_models()
medspacy_nlp = build_medspacy_pipeline()

# Test on one report
sample_text = df["transcription"].iloc[0]
print(f"\nReport snippet: {sample_text[:200]}\n")

entities = extract_entities(sample_text, nlp_sci, nlp_bc5, medspacy_nlp)
positive = get_positive_entities(entities)
deduped = deduplicate_entities(positive)

print(f"Total entities extracted: {len(entities)}")
print(f"After negation filter:    {len(positive)}")
print(f"After deduplication:      {len(deduped)}")

print("\n=== ENTITIES ===")
for ent in deduped:
    flag = ""
    if ent["uncertain"]: flag += " [UNCERTAIN]"
    if ent["family"]: flag += " [FAMILY]"
    print(f"  [{ent['label']:10}] {ent['text']}{flag}  (source: {ent['source']})")

# Sanity check against keywords on 10 reports
extract_and_sanity_check(df, nlp_sci, nlp_bc5, medspacy_nlp, n_samples=10)