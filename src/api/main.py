# src/api/main.py
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
from loguru import logger
from src.api.inference import get_models, run_stage1, run_stage2, run_ner_pipeline

# ── Request / Response models ─────────────────────────────────────────────────

class ReportRequest(BaseModel):
    text: str
    include_shap: bool = True

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v):
        v = v.strip()
        if len(v) < 20:
            raise ValueError("Report text must be at least 20 characters.")
        if len(v) > 50000:
            raise ValueError("Report text exceeds maximum length of 50,000 characters.")
        return v


class DoctypeResponse(BaseModel):
    predicted_label: str
    confidence: float
    all_scores: dict
    token_count: int
    over_512_warning: bool
    warning_message: str | None
    n_chunks_used: int
    processing_time_ms: float


class SpecialtyResponse(BaseModel):
    stage1_label: str
    stage1_confidence: float
    routed_to_stage2: bool
    stage2_label: str | None
    stage2_confidence: float | None
    stage2_all_scores: dict | None
    top_shap_tokens: list | None
    shap_html: str | None
    token_count: int
    over_512_warning: bool
    warning_message: str | None
    processing_time_ms: float


class FullResponse(BaseModel):
    stage1_label: str
    stage1_confidence: float
    routed_to_stage2: bool
    stage2_label: str | None
    stage2_confidence: float | None
    sections: list
    entities: list
    icd10_codes: list
    entity_count: dict
    top_shap_tokens: list | None
    shap_html: str | None
    token_count: int
    over_512_warning: bool
    warning_message: str | None
    processing_time_ms: float


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load all models on startup so first request isn't slow."""
    logger.info("Starting up — loading all models into memory...")
    get_models()
    logger.info("All models ready. API is live.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Clinical NLP Pipeline API",
    description=(
        "Two-stage hierarchical clinical document classification. "
        "Stage 1 identifies document type; Stage 2 classifies clinical specialty. "
        "Includes NER, ICD-10 mapping, and SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "device": "cuda" if __import__("torch").cuda.is_available() else "cpu"}


@app.get("/metrics")
def metrics():
    """Model performance metrics for monitoring."""
    return {
        "stage1": {
            "model":      "Bio_ClinicalBERT + FocalLoss + structural features",
            "test_macro_f1": 0.528,
            "baseline_f1":   0.494,
            "n_classes":  5,
        },
        "stage2": {
            "model":      "Bio_ClinicalBERT chunk-and-pool",
            "test_macro_f1": 0.662,
            "baseline_f1":   0.605,
            "n_classes":  11,
        },
    }


@app.post("/predict/doctype", response_model=DoctypeResponse)
def predict_doctype(request: ReportRequest):
    """
    Stage 1 only — classify the document type of a clinical report.
    Returns one of: consultation, discharge_summary, procedure_note,
    progress_note, specialty_report.
    """
    t0 = time.time()
    try:
        models = get_models()
        result = run_stage1(request.text, models)
        result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
        return result
    except Exception as e:
        logger.error(f"/predict/doctype error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/specialty", response_model=SpecialtyResponse)
def predict_specialty(request: ReportRequest):
    """
    Two-stage classification.
    Routes through Stage 1 first — only specialty_report documents
    continue to Stage 2 for specialty classification.
    Returns SHAP explanation for Stage 2 predictions.
    """
    t0 = time.time()
    try:
        models  = get_models()
        s1_result = run_stage1(request.text, models)
        s1_label  = s1_result["predicted_label"]
        routed    = s1_label == "specialty_report"

        s2_label = s2_conf = s2_scores = top_tokens = shap_html = None

        if routed:
            s2_result = run_stage2(
                request.text, models,
                include_shap=request.include_shap
            )
            s2_label   = s2_result["predicted_label"]
            s2_conf    = s2_result["confidence"]
            s2_scores  = s2_result["all_scores"]
            if request.include_shap and "shap_explanation" in s2_result:
                top_tokens = s2_result["shap_explanation"]["top_tokens"]
                shap_html  = s2_result["shap_html"]

        return SpecialtyResponse(
            stage1_label=s1_label,
            stage1_confidence=s1_result["confidence"],
            routed_to_stage2=routed,
            stage2_label=s2_label,
            stage2_confidence=s2_conf,
            stage2_all_scores=s2_scores,
            top_shap_tokens=top_tokens,
            shap_html=shap_html,
            token_count=s1_result["token_count"],
            over_512_warning=s1_result["over_512_warning"],
            processing_time_ms=round((time.time() - t0) * 1000, 1),
        )
    except Exception as e:
        logger.error(f"/predict/specialty error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/full", response_model=FullResponse)
def predict_full(request: ReportRequest):
    """
    Full pipeline — Stage 1 + Stage 2 + NER + ICD-10 + SHAP.
    The complete end-to-end inference for a single clinical report.
    """
    t0 = time.time()
    try:
        models    = get_models()
        s1_result = run_stage1(request.text, models)
        s1_label  = s1_result["predicted_label"]
        routed    = s1_label == "specialty_report"
        ner_result = run_ner_pipeline(request.text, models)

        s2_label = s2_conf = top_tokens = shap_html = None

        if routed:
            s2_result = run_stage2(
                request.text, models,
                include_shap=request.include_shap
            )
            s2_label = s2_result["predicted_label"]
            s2_conf  = s2_result["confidence"]
            if request.include_shap and "shap_explanation" in s2_result:
                top_tokens = s2_result["shap_explanation"]["top_tokens"]
                shap_html  = s2_result["shap_html"]

        return FullResponse(
            stage1_label=s1_label,
            stage1_confidence=s1_result["confidence"],
            routed_to_stage2=routed,
            stage2_label=s2_label,
            stage2_confidence=s2_conf,
            sections=ner_result["sections"],
            entities=ner_result["entities"],
            icd10_codes=ner_result["icd10_codes"],
            entity_count=ner_result["entity_count"],
            top_shap_tokens=top_tokens,
            shap_html=shap_html,
            token_count=s1_result["token_count"],
            over_512_warning=s1_result["over_512_warning"],
            processing_time_ms=round((time.time() - t0) * 1000, 1),
        )
    except Exception as e:
        logger.error(f"/predict/full error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/{stage}", response_class=HTMLResponse)
def explain_html(stage: int, text: str):
    """
    GET endpoint that returns the SHAP HTML overlay directly in the browser.
    Useful for quick visual inspection during development.
    """
    if stage not in (1, 2):
        raise HTTPException(status_code=400, detail="Stage must be 1 or 2.")
    models = get_models()
    model  = models[f"s{stage}_model"]
    tok    = models[f"s{stage}_tok"]
    i2l    = models[f"s{stage}_i2l"]
    expl   = get_shap_explanation(text, model, tok, i2l, stage=stage)
    from src.explainability.shap_explainer import render_html_explanation
    return render_html_explanation(expl)