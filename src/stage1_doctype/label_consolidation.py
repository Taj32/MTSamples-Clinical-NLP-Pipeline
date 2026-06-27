import pandas as pd
from collections import Counter

# ── Stage 1 labels: document type ────────────────────────────────────────────
# These are note FORMAT labels, not clinical specialties
DOCUMENT_TYPE_MAP = {
    "Surgery":                          "procedure_note",
    "Consult - History and Phy.":       "consultation",
    "SOAP / Chart / Progress Notes":    "progress_note",
    "Discharge Summary":                "discharge_summary",
    "Office Notes":                     "progress_note",
    "Emergency Room Reports":           "progress_note",
}

# ── Stage 2 labels: clinical specialty ───────────────────────────────────────
# Only applies to rows NOT in DOCUMENT_TYPE_MAP
SPECIALTY_MAP = {
    # Cardiovascular
    "Cardiovascular / Pulmonary":       "cardiovascular_pulmonary",

    # Orthopedic / Musculoskeletal
    "Orthopedic":                       "orthopedic",
    "Podiatry":                         "orthopedic",

    # Neurology
    "Neurology":                        "neurology",
    "Neurosurgery":                     "neurology",

    # Gastroenterology
    "Gastroenterology":                 "gastroenterology",

    # Urology / Nephrology
    "Urology":                          "urology_nephrology",
    "Nephrology":                       "urology_nephrology",

    # OB/GYN
    "Obstetrics / Gynecology":          "obgyn",

    # ENT / Ophthalmology
    "ENT - Otolaryngology":             "ent_ophthalmology",
    "Ophthalmology":                    "ent_ophthalmology",

    # Psychiatry
    "Psychiatry / Psychology":          "psychiatry",

    # Oncology
    "Hematology - Oncology":            "oncology",

    # Radiology — kept separate, strong lexical signal
    "Radiology":                        "radiology",

    # General / Other — catch-all for small classes
    "General Medicine":                 "general_other",
    "Pediatrics - Neonatal":            "general_other",
    "Pain Management":                  "general_other",
    "Dermatology":                      "general_other",
    "Allergy / Immunology":             "general_other",
    "Rheumatology":                     "general_other",
    "Endocrinology":                    "general_other",
    "Hospice - Palliative Care":        "general_other",
    "Physical Medicine - Rehab":        "general_other",
    "Sleep Medicine":                   "general_other",
    "Chiropractic":                     "general_other",
    "Dentistry":                        "general_other",
    "Diets and Nutritions":             "general_other",
    "IME-QME-Work Comp etc.":           "general_other",
    "Lab Medicine - Pathology":         "general_other",
    "Letters":                          "general_other",
    "Autopsy":                          "general_other",
    "Speech - Language":                "general_other",
    "Cosmetic / Plastic Surgery":       "general_other",
    "Bariatrics":                       "general_other",
}


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two new columns to the dataframe:
      - stage1_label: document type (for all rows)
      - stage2_label: clinical specialty (None for document-type rows)
      - is_specialty_report: bool flag for Stage 2 training
    """
    df = df.copy()

    def get_stage1(specialty):
        return DOCUMENT_TYPE_MAP.get(specialty.strip(), "specialty_report")

    def get_stage2(specialty):
        specialty = specialty.strip()
        if specialty in DOCUMENT_TYPE_MAP:
            return None
        return SPECIALTY_MAP.get(specialty, "general_other")

    df["stage1_label"] = df["medical_specialty"].apply(get_stage1)
    df["stage2_label"] = df["medical_specialty"].apply(get_stage2)
    df["is_specialty_report"] = df["stage1_label"] == "specialty_report"

    return df


def print_label_distribution(df: pd.DataFrame) -> None:
    """Print class distributions for both stages."""
    print("\n" + "="*60)
    print("STAGE 1 — Document Type Distribution")
    print("="*60)
    s1 = df["stage1_label"].value_counts()
    for label, count in s1.items():
        pct = count / len(df) * 100
        print(f"  {label:25} {count:5d}  ({pct:.1f}%)")

    print("\n" + "="*60)
    print("STAGE 2 — Specialty Distribution (specialty reports only)")
    print("="*60)
    specialty_df = df[df["is_specialty_report"]]
    s2 = specialty_df["stage2_label"].value_counts()
    for label, count in s2.items():
        pct = count / len(specialty_df) * 100
        print(f"  {label:25} {count:5d}  ({pct:.1f}%)")

    print(f"\n  Total specialty reports: {len(specialty_df)}")
    print(f"  Total document-type reports: {len(df) - len(specialty_df)}")
    print(f"  Total reports: {len(df)}")


def get_class_weights(labels: pd.Series) -> dict:
    """
    Compute inverse-frequency class weights for imbalance handling.
    Returns dict mapping label -> weight.
    """
    counts = Counter(labels)
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {
        label: total / (n_classes * count)
        for label, count in counts.items()
    }
    return weights