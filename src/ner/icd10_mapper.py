import json
import re
from pathlib import Path
from loguru import logger

# Lightweight ICD-10 lookup table for common conditions found in MTSamples
# Format: {normalized_term: (icd10_code, description)}
ICD10_LOOKUP = {
    # Cardiovascular
    "congestive heart failure": ("I50.9", "Heart failure, unspecified"),
    "heart failure": ("I50.9", "Heart failure, unspecified"),
    "hypertension": ("I10", "Essential hypertension"),
    "atrial fibrillation": ("I48.91", "Unspecified atrial fibrillation"),
    "myocardial infarction": ("I21.9", "Acute myocardial infarction, unspecified"),
    "coronary artery disease": ("I25.10", "Atherosclerotic heart disease"),
    "aortic stenosis": ("I35.0", "Nonrheumatic aortic stenosis"),
    "mitral regurgitation": ("I34.0", "Nonrheumatic mitral valve regurgitation"),
    "chest pain": ("R07.9", "Chest pain, unspecified"),
    "shortness of breath": ("R06.00", "Dyspnea, unspecified"),
    "dyspnea": ("R06.00", "Dyspnea, unspecified"),

    # Pulmonary
    "pneumonia": ("J18.9", "Pneumonia, unspecified organism"),
    "asthma": ("J45.909", "Unspecified asthma, uncomplicated"),
    "copd": ("J44.1", "COPD with acute exacerbation"),
    "chronic obstructive pulmonary disease": ("J44.1", "COPD with acute exacerbation"),
    "pulmonary embolism": ("I26.99", "Other pulmonary embolism"),
    "pleural effusion": ("J91.8", "Pleural effusion in other conditions"),
    "upper respiratory infection": ("J06.9", "Acute upper respiratory infection"),

    # Neurological
    "stroke": ("I63.9", "Cerebral infarction, unspecified"),
    "cerebrovascular accident": ("I63.9", "Cerebral infarction, unspecified"),
    "headache": ("R51.9", "Headache, unspecified"),
    "migraine": ("G43.909", "Migraine, unspecified, not intractable"),
    "seizure": ("R56.9", "Unspecified convulsions"),
    "epilepsy": ("G40.909", "Epilepsy, unspecified"),
    "parkinson": ("G20", "Parkinson's disease"),
    "alzheimer": ("G30.9", "Alzheimer's disease, unspecified"),
    "multiple sclerosis": ("G35", "Multiple sclerosis"),

    # Gastrointestinal
    "gastroesophageal reflux": ("K21.0", "GERD with esophagitis"),
    "gerd": ("K21.0", "GERD with esophagitis"),
    "peptic ulcer": ("K27.9", "Peptic ulcer, unspecified"),
    "appendicitis": ("K37", "Unspecified appendicitis"),
    "cholecystitis": ("K81.9", "Cholecystitis, unspecified"),
    "crohn": ("K50.90", "Crohn's disease, unspecified"),
    "colitis": ("K51.90", "Ulcerative colitis, unspecified"),
    "irritable bowel": ("K58.9", "Irritable bowel syndrome"),

    # Endocrine / Metabolic
    "diabetes mellitus": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "diabetes": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "obesity": ("E66.9", "Obesity, unspecified"),
    "morbid obesity": ("E66.01", "Morbid obesity due to excess calories"),
    "hypothyroidism": ("E03.9", "Hypothyroidism, unspecified"),
    "hyperthyroidism": ("E05.90", "Hyperthyroidism, unspecified"),

    # Musculoskeletal
    "fracture": ("M84.40", "Pathological fracture, unspecified site"),
    "osteoarthritis": ("M19.90", "Unspecified osteoarthritis"),
    "rheumatoid arthritis": ("M06.9", "Rheumatoid arthritis, unspecified"),
    "back pain": ("M54.50", "Low back pain, unspecified"),
    "osteoporosis": ("M81.0", "Age-related osteoporosis"),

    # Oncology
    "carcinoma": ("C80.1", "Malignant neoplasm, unspecified"),
    "lymphoma": ("C85.90", "Non-Hodgkin lymphoma, unspecified"),
    "melanoma": ("C43.9", "Malignant melanoma of skin, unspecified"),
    "leukemia": ("C95.90", "Leukemia, unspecified"),

    # Renal
    "chronic kidney disease": ("N18.9", "Chronic kidney disease, unspecified"),
    "urinary tract infection": ("N39.0", "Urinary tract infection"),
    "uti": ("N39.0", "Urinary tract infection"),
    "kidney stones": ("N20.0", "Calculus of kidney"),

    # Psychiatric
    "depression": ("F32.9", "Major depressive disorder, unspecified"),
    "anxiety": ("F41.9", "Anxiety disorder, unspecified"),
    "bipolar": ("F31.9", "Bipolar disorder, unspecified"),
    "schizophrenia": ("F20.9", "Schizophrenia, unspecified"),
    "ptsd": ("F43.10", "Post-traumatic stress disorder, unspecified"),

    # Allergic / Immunologic
    "allergic rhinitis": ("J30.9", "Allergic rhinitis, unspecified"),
    "anaphylaxis": ("T78.2XXA", "Anaphylactic shock, unspecified"),

    # Dermatologic
    "eczema": ("L30.9", "Dermatitis, unspecified"),
    "psoriasis": ("L40.9", "Psoriasis, unspecified"),
    "cellulitis": ("L03.90", "Cellulitis, unspecified"),
}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def map_entity_to_icd10(entity_text: str) -> dict | None:
    """
    Attempt to map an entity string to an ICD-10 code.
    Returns dict with code and description, or None if no match.
    Uses substring matching for flexibility.
    """
    normalized = normalize_text(entity_text)

    # Exact match first
    if normalized in ICD10_LOOKUP:
        code, desc = ICD10_LOOKUP[normalized]
        return {"code": code, "description": desc, "match_type": "exact"}

    # Substring match — check if any lookup key appears in the entity text
    for key, (code, desc) in ICD10_LOOKUP.items():
        if key in normalized:
            return {"code": code, "description": desc, "match_type": "substring"}

    # Reverse substring — check if entity text appears in any lookup key
    for key, (code, desc) in ICD10_LOOKUP.items():
        if normalized in key and len(normalized) > 4:
            return {"code": code, "description": desc, "match_type": "partial"}

    return None


def build_report_json(
    report_index: int,
    specialty: str,
    original_text: str,
    entities: list[dict],
    sections: dict,
) -> dict:
    """
    Build the structured JSON output for a single report.
    This is the audit trail stored per report for downstream use.
    """
    mapped_entities = []
    for ent in entities:
        icd = map_entity_to_icd10(ent["text"])
        mapped_entities.append({
            "text": ent["text"],
            "label": ent["label"],
            "source": ent["source"],
            "negated": ent["negated"],
            "uncertain": ent["uncertain"],
            "family": ent["family"],
            "icd10": icd,
        })

    # Separate positive vs contextual entities
    positive = [e for e in mapped_entities if not e["negated"] and not e["family"]]
    icd_coded = [e for e in positive if e["icd10"] is not None]

    return {
        "report_id": report_index,
        "specialty": specialty,
        "sections_found": list(sections.keys()),
        "entity_count": {
            "total": len(mapped_entities),
            "positive": len(positive),
            "icd_coded": len(icd_coded),
        },
        "entities": mapped_entities,
        "icd10_codes": [
            {
                "code": e["icd10"]["code"],
                "description": e["icd10"]["description"],
                "entity": e["text"],
                "match_type": e["icd10"]["match_type"],
            }
            for e in icd_coded
        ],
    }


def process_dataset(df, nlp_sci, nlp_bc5, medspacy_nlp,
                    output_path: str = "data/processed/ner_output.jsonl",
                    n_samples: int = None) -> None:
    """
    Run the full NER + ICD-10 pipeline over the dataset.
    Saves one JSON object per line (JSONL format) for easy streaming.
    """
    from src.ner.entity_extractor import (
        extract_entities, get_positive_entities, deduplicate_entities
    )
    from src.preprocessing.medspacy_pipeline import preprocess_report

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset = df.head(n_samples) if n_samples else df
    logger.info(f"Processing {len(subset)} reports...")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in subset.iterrows():
            try:
                text = row["transcription"]
                specialty = row["medical_specialty"].strip()

                # Preprocess
                preprocessed = preprocess_report(text, medspacy_nlp)

                # Extract entities
                entities = extract_entities(
                    preprocessed["expanded"],
                    nlp_sci, nlp_bc5, medspacy_nlp
                )
                positive = get_positive_entities(entities)
                deduped = deduplicate_entities(positive)

                # Build JSON record
                record = build_report_json(
                    report_index=int(idx),
                    specialty=specialty,
                    original_text=text,
                    entities=deduped,
                    sections=preprocessed["sections"],
                )

                f.write(json.dumps(record) + "\n")

            except Exception as e:
                logger.warning(f"Failed on report {idx}: {e}")
                continue

    logger.info(f"Saved NER output to {output_path}")