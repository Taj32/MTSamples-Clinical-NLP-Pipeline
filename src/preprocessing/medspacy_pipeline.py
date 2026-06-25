import medspacy
from loguru import logger
logger.disable("PyRuSH")


# Clinical abbreviation expansion dictionary
ABBREV_MAP = {
    "SOB": "shortness of breath",
    "CHF": "congestive heart failure",
    "f/u": "follow up",
    "hx": "history",
    "h/o": "history of",
    "Hx": "history",
    "dx": "diagnosis",
    "Dx": "diagnosis",
    "sx": "symptoms",
    "rx": "prescription",
    "Rx": "prescription",
    "pt": "patient",
    "yo": "year old",
    "y/o": "year old",
    "c/o": "complains of",
    "w/o": "without",
    "s/p": "status post",
    "PMH": "past medical history",
    "FH": "family history",
    "SH": "social history",
    "ROS": "review of systems",
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "GERD": "gastroesophageal reflux disease",
    "COPD": "chronic obstructive pulmonary disease",
    "MI": "myocardial infarction",
    "CVA": "cerebrovascular accident",
    "URI": "upper respiratory infection",
    "UTI": "urinary tract infection",
    "BP": "blood pressure",
    "HR": "heart rate",
    "RR": "respiratory rate",
    "Temp": "temperature",
    "WBC": "white blood cell",
    "RBC": "red blood cell",
    "Hgb": "hemoglobin",
    "BMP": "basic metabolic panel",
    "CBC": "complete blood count",
    "EKG": "electrocardiogram",
    "ECG": "electrocardiogram",
    "MRI": "magnetic resonance imaging",
    "CT": "computed tomography",
    "IV": "intravenous",
    "PO": "by mouth",
    "PRN": "as needed",
    "QD": "once daily",
    "BID": "twice daily",
    "TID": "three times daily",
    "QID": "four times daily",
}

# Section header patterns for MTSamples
SECTION_PATTERNS = [
    "chief complaint",
    "history of present illness",
    "past medical history",
    "past surgical history",
    "family history",
    "social history",
    "review of systems",
    "physical examination",
    "vital signs",
    "assessment",
    "plan",
    "assessment and plan",
    "medications",
    "allergies",
    "laboratory",
    "impression",
    "findings",
    "procedure",
    "diagnosis",
    "discharge instructions",
]


def expand_abbreviations(text: str) -> str:
    """Replace clinical abbreviations with full forms."""
    for abbrev, expansion in ABBREV_MAP.items():
        # Word-boundary replacement to avoid partial matches
        import re
        text = re.sub(rf'\b{re.escape(abbrev)}\b', expansion, text)
    return text


def build_medspacy_pipeline():
    """
    Build and return a medspaCy pipeline with:
    - PyRuSH sentence segmentation
    - ConText for negation, uncertainty, family history
    """
    nlp = medspacy.load(enable=["medspacy_pyrush", "medspacy_context"])
    return nlp


def parse_sections(text: str) -> dict:
    """
    Extract sections from a clinical note based on header patterns.
    Returns dict of {section_name: section_text}.
    """
    import re
    sections = {}
    # Build pattern that matches any section header
    pattern = "|".join(
        rf"(?P<{re.sub('[^a-z0-9]', '_', s)}>{re.escape(s)}[\s]*:)"
        for s in SECTION_PATTERNS
    )
    splits = re.split(
        r'(' + '|'.join(rf'{re.escape(s)}\s*:' for s in SECTION_PATTERNS) + r')',
        text,
        flags=re.IGNORECASE
    )

    current_section = "preamble"
    sections[current_section] = ""
    i = 0
    while i < len(splits):
        chunk = splits[i]
        matched_section = next(
            (s for s in SECTION_PATTERNS
             if chunk.strip().lower().rstrip(":") == s),
            None
        )
        if matched_section:
            current_section = matched_section
            sections[current_section] = ""
        else:
            sections[current_section] = sections.get(current_section, "") + chunk
        i += 1

    # Strip whitespace from all sections
    return {k: v.strip() for k, v in sections.items() if v.strip()}


def preprocess_report(text: str, nlp) -> dict:
    """
    Full preprocessing pipeline for a single MTSamples report.
    Returns structured dict with expanded text, sections, and spaCy doc.
    """
    # Step 1: abbreviation expansion
    expanded = expand_abbreviations(text)

    # Step 2: section parsing
    sections = parse_sections(expanded)

    # Step 3: run medspaCy for sentence segmentation + ConText
    doc = nlp(expanded)

    return {
        "original": text,
        "expanded": expanded,
        "sections": sections,
        "doc": doc,
    }