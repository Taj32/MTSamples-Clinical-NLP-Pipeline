import spacy
import pandas as pd
from loguru import logger
logger.disable("PyRuSH")

from src.preprocessing.medspacy_pipeline import build_medspacy_pipeline, preprocess_report


# Entity labels we care about from each model
SCI_LABELS_OF_INTEREST = {"ENTITY"}  # en_core_sci_md uses generic ENTITY
BC5CDR_LABELS = {"DISEASE", "CHEMICAL"}

# Noise entities to filter out — too short or too generic to be useful
MIN_ENTITY_LENGTH = 3
STOPWORD_ENTITIES = {
    "patient", "history", "pain", "no", "the", "a", "an",
    "mg", "ml", "he", "she", "his", "her", "year", "old",
    "day", "days", "week", "weeks", "month", "months",
    "written", "summer", "short time", "effectiveness",
    "pleasant", "gentleman", "difficulty", "normal", "wall",
    "age", "floor", "abc", "past", "samples",
}

def load_ner_models():
    """Load both scispaCy NER models."""
    logger.info("Loading en_core_sci_md...")
    nlp_sci = spacy.load("en_core_sci_md")

    logger.info("Loading en_ner_bc5cdr_md...")
    nlp_bc5 = spacy.load("en_ner_bc5cdr_md")

    return nlp_sci, nlp_bc5


def is_valid_entity(ent_text: str, label: str = "") -> bool:
    """Filter out noise entities."""
    text = ent_text.strip().lower()
    if len(text) < MIN_ENTITY_LENGTH:
        return False
    if text in STOPWORD_ENTITIES:
        return False
    if text.isdigit():
        return False
    
     # Drop single lowercase words from the broad sci model — usually noise
    if label == "ENTITY" and len(text.split()) == 1 and text.islower():
        return False
    return True


def extract_entities(text: str, nlp_sci, nlp_bc5, medspacy_nlp) -> list[dict]:
    """
    Extract entities from a single report using both NER models.
    Applies negation filtering via medspaCy ConText.

    Returns list of dicts:
    {
        "text": str,
        "label": str,
        "source": str,       # "sci" or "bc5cdr"
        "negated": bool,
        "uncertain": bool,
        "family": bool,
        "start": int,
        "end": int,
    }
    """
    # Run medspaCy first to get ConText attributes on sentences
    med_doc = medspacy_nlp(text)

    # Build a set of negated spans from medspaCy
    # ConText works on entities — we'll cross-reference by character offsets
    negated_spans = set()
    uncertain_spans = set()
    family_spans = set()

    # Run scispaCy
    sci_doc = nlp_sci(text)
    bc5_doc = nlp_bc5(text)

    entities = []

    # Process scispaCy entities
    for ent in sci_doc.ents:
        if not is_valid_entity(ent.text, ent.label_):
            continue
        entities.append({
            "text": ent.text.strip(),
            "label": ent.label_,
            "source": "sci",
            "negated": False,      # will be resolved below
            "uncertain": False,
            "family": False,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    # Process BC5CDR entities
    for ent in bc5_doc.ents:
        if not is_valid_entity(ent.text, ent.label_):
            continue
        if ent.label_ not in BC5CDR_LABELS:
            continue
        entities.append({
            "text": ent.text.strip(),
            "label": ent.label_,
            "source": "bc5cdr",
            "negated": False,
            "uncertain": False,
            "family": False,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    # Apply ConText negation via medspaCy
    # medspaCy needs entities defined via target_matcher to apply ConText
    # We approximate by checking if entity text appears in a negated sentence context
    negated_phrases = set()
    uncertain_phrases = set()
    family_phrases = set()

    for sent in med_doc.sents:
        sent_text_lower = sent.text.lower()
        # Check for negation triggers around known entity text
        negation_triggers = ["no ", "not ", "without ", "denies ", "negative for "]
        uncertainty_triggers = ["possible ", "probable ", "suspected ", "rule out "]
        family_triggers = ["mother", "father", "family history", "sister", "brother",
                          "parent", "grandparent", "familial"]

        for ent in entities:
            ent_lower = ent["text"].lower()
            if ent_lower in sent_text_lower:
                for trigger in negation_triggers:
                    if trigger in sent_text_lower:
                        negated_phrases.add(ent_lower)
                for trigger in uncertainty_triggers:
                    if trigger in sent_text_lower:
                        uncertain_phrases.add(ent_lower)
                for trigger in family_triggers:
                    if trigger in sent_text_lower:
                        family_phrases.add(ent_lower)

    # Apply context flags to entities
    for ent in entities:
        ent_lower = ent["text"].lower()
        ent["negated"] = ent_lower in negated_phrases
        ent["uncertain"] = ent_lower in uncertain_phrases
        ent["family"] = ent_lower in family_phrases

    return entities


def get_positive_entities(entities: list[dict]) -> list[dict]:
    """Return only non-negated, non-family entities for classification use."""
    return [
        e for e in entities
        if not e["negated"] and not e["family"]
    ]


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Remove duplicate entity texts, keeping the most specific label."""
    seen = {}
    for ent in entities:
        key = ent["text"].lower()
        if key not in seen:
            seen[key] = ent
        else:
            # Prefer BC5CDR label (more specific) over generic ENTITY
            if ent["source"] == "bc5cdr":
                seen[key] = ent
    return list(seen.values())


def extract_and_sanity_check(df: pd.DataFrame,
                              nlp_sci, nlp_bc5, medspacy_nlp,
                              n_samples: int = 10) -> None:
    """
    Run NER on n_samples reports and sanity-check against
    the keywords column as rough ground truth.
    """
    print(f"\n{'='*60}")
    print(f"NER SANITY CHECK — {n_samples} samples")
    print(f"{'='*60}")

    sample = df.dropna(subset=["keywords"]).head(n_samples)

    for _, row in sample.iterrows():
        text = row["transcription"]
        keywords = [k.strip().lower() for k in str(row["keywords"]).split(",")]
        specialty = row["medical_specialty"].strip()

        entities = extract_entities(text, nlp_sci, nlp_bc5, medspacy_nlp)
        positive = get_positive_entities(entities)
        deduped = deduplicate_entities(positive)

        extracted_texts = [e["text"].lower() for e in deduped]

        # Check overlap with keywords
        hits = [kw for kw in keywords if any(kw in et or et in kw
                                              for et in extracted_texts)]
        coverage = len(hits) / len(keywords) if keywords else 0

        print(f"\nSpecialty: {specialty}")
        print(f"Keywords:  {keywords[:5]}")
        print(f"Extracted: {[e['text'] for e in deduped[:8]]}")
        print(f"Keyword coverage: {coverage:.0%} ({len(hits)}/{len(keywords)} matched)")