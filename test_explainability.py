# test_explainability.py
import torch
from src.explainability.shap_explainer import (
    load_model_from_checkpoint,
    predict_single,
    get_shap_explanation,
    format_explanation_text,
)

# Load both models
s1_model, s1_tokenizer, s1_label2id, s1_id2label = load_model_from_checkpoint(
    "models/stage1_clinicalbert_BEST_20260630_124540.pt", stage=1
)
s2_model, s2_tokenizer, s2_label2id, s2_id2label = load_model_from_checkpoint(
    "models/stage2_clinicalbert_BEST_oldest.pt", stage=2 
)

# Test report
sample = """SUBJECTIVE: This 45-year-old male presents with complaint of chest pain 
radiating to the left arm for the past 2 hours. He has a history of hypertension 
and hyperlipidemia. He denies shortness of breath. Current medications include 
lisinopril and atorvastatin. OBJECTIVE: BP 158/94, HR 88, RR 16. EKG shows 
ST elevation in leads II, III, aVF. ASSESSMENT: Acute inferior MI. 
PLAN: Transfer to cath lab immediately."""

print("=== STAGE 1 PREDICTION ===")
s1_pred = predict_single(sample, s1_model, s1_tokenizer, s1_id2label, stage=1)
print(f"Document type: {s1_pred['predicted_label']} ({s1_pred['confidence']:.1%})")
print(f"All scores: {s1_pred['all_scores']}")
print(f"Token count: {s1_pred['token_count']} | Over 512: {s1_pred['over_512_warning']}")

print("\n=== STAGE 2 PREDICTION ===")
s2_pred = predict_single(sample, s2_model, s2_tokenizer, s2_id2label, stage=2)
print(f"Specialty: {s2_pred['predicted_label']} ({s2_pred['confidence']:.1%})")
print(f"All scores: {s2_pred['all_scores']}")

print("\n=== SHAP EXPLANATION (Stage 2) ===")
print("Computing... this takes ~60 seconds")
explanation = get_shap_explanation(
    sample, s2_model, s2_tokenizer, s2_id2label, stage=2, n_samples=50
)
print(format_explanation_text(explanation))