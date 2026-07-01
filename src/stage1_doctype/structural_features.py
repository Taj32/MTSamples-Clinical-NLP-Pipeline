import re
import numpy as np

# Strong structural markers for each document type
PROCEDURE_MARKERS = [
    "preoperative diagnosis", "postoperative diagnosis", "procedure performed",
    "anesthesia:", "estimated blood loss", "indications for procedure",
    "operative report", "surgeon:", "assistant surgeon",
]
CONSULT_MARKERS = [
    "reason for consultation", "history of present illness",
    "chief complaint", "consulted for", "referring physician",
]
DISCHARGE_MARKERS = [
    "discharge diagnosis", "discharge medications", "discharge instructions",
    "condition on discharge", "follow-up instructions", "hospital course",
]
PROGRESS_MARKERS = [
    "subjective:", "objective:", "assessment:", "plan:",
    "soap note", "progress note",
]

PAST_TENSE_VERBS = [
    "was performed", "was made", "was prepped", "was draped", "was placed",
    "was administered", "was identified", "was removed", "was closed",
]
PRESENT_TENSE_MARKERS = [
    "presents with", "continues to", "complains of", "reports",
    "is a", "denies", "endorses",
]


def extract_structural_features(text: str) -> np.ndarray:
    """
    Extract a small set of structural/formatting features that
    distinguish document TYPE independent of clinical content.

    Returns a fixed-length numpy array of 7 features:
    [procedure_score, consult_score, discharge_score, progress_score,
     past_tense_ratio, has_numbered_list, doc_length_bucket]
    """
    text_lower = text.lower()

    def marker_score(markers):
        return sum(1 for m in markers if m in text_lower) / len(markers)

    procedure_score = marker_score(PROCEDURE_MARKERS)
    consult_score   = marker_score(CONSULT_MARKERS)
    discharge_score = marker_score(DISCHARGE_MARKERS)
    progress_score  = marker_score(PROGRESS_MARKERS)

    past_count = sum(1 for p in PAST_TENSE_VERBS if p in text_lower)
    present_count = sum(1 for p in PRESENT_TENSE_MARKERS if p in text_lower)
    total_tense_signals = past_count + present_count
    past_tense_ratio = past_count / total_tense_signals if total_tense_signals > 0 else 0.5

    has_numbered_list = 1.0 if re.search(r'\b\d\.\s', text) else 0.0

    # Document length bucket, normalized 0-1 (cap at 3000 chars)
    doc_length_bucket = min(len(text) / 3000, 1.0)

    return np.array([
        procedure_score, consult_score, discharge_score, progress_score,
        past_tense_ratio, has_numbered_list, doc_length_bucket
    ], dtype=np.float32)


N_STRUCTURAL_FEATURES = 7