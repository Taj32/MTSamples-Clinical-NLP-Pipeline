# icd10_test.py
import pandas as pd
import json
from src.preprocessing.medspacy_pipeline import build_medspacy_pipeline
from src.ner.entity_extractor import load_ner_models
from src.ner.icd10_mapper import process_dataset, map_entity_to_icd10

# Quick unit test on the mapper itself
test_terms = [
    "congestive heart failure", "shortness of breath",
    "allergic rhinitis", "morbid obesity", "pneumonia"
]
print("=== ICD-10 MAPPING TEST ===")
for term in test_terms:
    result = map_entity_to_icd10(term)
    print(f"  {term:35} -> {result}")

# Run pipeline on 20 reports
df = pd.read_csv("data/processed/mtsamples_clean.csv", index_col=0)
nlp_sci, nlp_bc5 = load_ner_models()
medspacy_nlp = build_medspacy_pipeline()

process_dataset(df, nlp_sci, nlp_bc5, medspacy_nlp,
                output_path="data/processed/ner_output.jsonl",
                n_samples=20)

# Preview first record
print("\n=== SAMPLE OUTPUT RECORD ===")
with open("data/processed/ner_output.jsonl") as f:
    record = json.loads(f.readline())
    print(f"Report ID:    {record['report_id']}")
    print(f"Specialty:    {record['specialty']}")
    print(f"Sections:     {record['sections_found']}")
    print(f"Entity counts: {record['entity_count']}")
    print(f"\nICD-10 codes found:")
    for code in record["icd10_codes"]:
        print(f"  {code['code']} | {code['description']:40} | from: '{code['entity']}'")
        
        
print("=============================================================================")
