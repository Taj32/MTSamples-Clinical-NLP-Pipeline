# src/api/inference.py
import torch
from functools import lru_cache
from loguru import logger
from src.explainability.shap_explainer import (
    load_model_from_checkpoint,
    predict_single,
    get_shap_explanation,
    render_html_explanation,
)
from src.preprocessing.medspacy_pipeline import build_medspacy_pipeline, preprocess_report
from src.ner.entity_extractor import (
    load_ner_models, extract_entities,
    get_positive_entities, deduplicate_entities
)
from src.ner.icd10_mapper import build_report_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoint paths — update if filenames differ
STAGE1_CHECKPOINT = "models/stage1_clinicalbert_BEST_20260630_124540.pt"
STAGE2_CHECKPOINT = "models/stage2_clinicalbert_BEST_oldest.pt"


@lru_cache(maxsize=1)
def get_models():
    """
    Load all models once and cache them.
    lru_cache ensures this only runs on the first request — not on every call.
    """
    logger.info("Loading Stage 1 model...")
    s1_model, s1_tok, s1_l2i, s1_i2l = load_model_from_checkpoint(
        STAGE1_CHECKPOINT, stage=1
    )

    logger.info("Loading Stage 2 model...")
    s2_model, s2_tok, s2_l2i, s2_i2l = load_model_from_checkpoint(
        STAGE2_CHECKPOINT, stage=2
    )

    logger.info("Loading NER models...")
    nlp_sci, nlp_bc5 = load_ner_models()

    logger.info("Loading medspaCy pipeline...")
    medspacy_nlp = build_medspacy_pipeline()

    logger.info("All models loaded and ready.")
    return {
        "s1_model": s1_model, "s1_tok": s1_tok,
        "s1_l2i": s1_l2i,     "s1_i2l": s1_i2l,
        "s2_model": s2_model, "s2_tok": s2_tok,
        "s2_l2i": s2_l2i,     "s2_i2l": s2_i2l,
        "nlp_sci": nlp_sci,   "nlp_bc5": nlp_bc5,
        "medspacy_nlp": medspacy_nlp,
    }


def run_stage1(text: str, models: dict) -> dict:
    """Run Stage 1 document type classification."""
    return predict_single(
        text, models["s1_model"], models["s1_tok"],
        models["s1_i2l"], stage=1
    )


def run_stage2(text: str, models: dict, include_shap: bool = True) -> dict:
    """Run Stage 2 specialty classification with optional SHAP explanation."""
    prediction = predict_single(
        text, models["s2_model"], models["s2_tok"],
        models["s2_i2l"], stage=2
    )
    if include_shap:
        explanation = get_shap_explanation(
            text, models["s2_model"], models["s2_tok"],
            models["s2_i2l"], stage=2
        )
        prediction["shap_explanation"] = explanation
        prediction["shap_html"] = render_html_explanation(explanation)
    return prediction


def run_ner_pipeline(text: str, models: dict) -> dict:
    """Run full NER + ICD-10 pipeline on text."""
    preprocessed = preprocess_report(text, models["medspacy_nlp"])
    entities = extract_entities(
        preprocessed["expanded"],
        models["nlp_sci"], models["nlp_bc5"],
        models["medspacy_nlp"]
    )
    positive = get_positive_entities(entities)
    deduped  = deduplicate_entities(positive)
    record   = build_report_json(
        report_index=0,
        specialty="unknown",
        original_text=text,
        entities=deduped,
        sections=preprocessed["sections"],
    )
    return {
        "sections":    list(preprocessed["sections"].keys()),
        "entities":    record["entities"],
        "icd10_codes": record["icd10_codes"],
        "entity_count": record["entity_count"],
    }