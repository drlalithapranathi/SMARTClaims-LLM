"""Remove radiology/imaging report sections from MIMIC-IV discharge summaries.

Builds a header classifier from a section list + normalization map, parses each
note into "HEADER:\\n..." sections, drops those classified as radiology (a SAFE
allowlist overrides removal), and reassembles. Defaults to KEEP — conservative,
since the output feeds a training corpus. Requires MIMIC-IV access via PhysioNet.
"""

import csv
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

SECTION_LIST_PATH = "section_list.csv"
HEADER_MAPPING_PATH = "section-header-mapping.csv"
INPUT_NOTES_PATH = "data/mimic-iv-note/discharge.csv"
OUTPUT_CLEANED_PATH = "output/cleaned_discharge_notes.csv"
OUTPUT_LOG_PATH = "output/removal_log.csv"

SAMPLE_SIZE = None          # int = random sample, None = all notes
RANDOM_SEED = 42
RADIOLOGY_CATEGORIES = ["Echo", "ECG", "Radiology"]

# Word boundaries keep \bct\b from firing inside "expect"; "dexa scan/study"
# avoids matching the medication "dexamethasone".
MODALITY_PATTERN = re.compile(
    r"(?i)\bct\b|\bmri\b|\bmra\b|\bcxr\b|\bxray\b|\bx-ray\b|\bx ray\b|"
    r"\bultrasound\b|\becho(?:cardiog)?\b|\bdoppler\b|\bangio\b|\bekg\b|\becg\b|\beeg\b|"
    r"\bfluoro(?:scopy)?\b|\bnuclear\b|\bradiograph\b|\bmammogra\w*\b|"
    r"\bpet scan\b|\bpet/|\bdexa scan\b|\bdexa study\b|\bbone scan\b|"
    r"\bcatheterization report\b|\bcath report\b|\bchest pa\b|\bchest ap\b|"
    r"\babdomen film\b|\bkub\b|\bvenogram\b|\barteriogram\b"
)
SECTION_PATTERN = re.compile(
    r"(?i)\bimpression\b|\bfindings\b|\btechnique\b|\bcomparison\b|\bindication\b|"
    r"\bwet read\b|\bmedical condition\b|\breason for this examination\b|"
    r"\bclinical history\b|\bclinical implications\b"
)
FALLBACK_RADIOLOGY_PATTERN = re.compile(
    r"(?i)\bct\b|\bmri\b|\bmra\b|\bcxr\b|\bxray\b|\bx-ray\b|\bx ray\b|"
    r"\bultrasound\b|\becho(?:cardiog)?\b|\bdoppler\b|\bangio(?:gram|graphy)?\b|"
    r"\bekg\b|\becg\b|\beeg\b|\bfluoro(?:scopy)?\b|\bnuclear\b|\bradiograph\b|"
    r"\bmammogra\w*\b|\bpet scan\b|\bdexa scan\b|\bdexa study\b|\bbone scan\b|"
    r"\bimpression\b|\bfindings\b|\btechnique\b|\bcomparison\b|\bindication\b|"
    r"\bwet read\b|\breason for this examination\b|\bmedical condition\b|"
    r"\bclinical history\b|\bclinical implications\b"
)

# A: single newline, uppercase, single line <=80 chars. B: double newline.
PATTERN_A = re.compile(r"\n([A-Z][A-Za-z\s&/,()\-]{0,79}?):\s*\n")
PATTERN_B = re.compile(r"\n\n([A-Za-z0-9 ]{1,80}):\s*\n")

SAFE_CANONICAL_HEADERS = {
    "hospital course", "brief hospital course", "history of present illness",
    "history of the present illness", "hpi",
    "discharge diagnosis", "discharge diagnoses", "admission diagnosis",
    "admission diagnoses", "principal diagnosis", "secondary diagnoses",
    "chief complaint", "chief complaints", "reason for admission",
    "diagnosis", "diagnoses",
    "physical examination", "physical exam", "admission physical exam",
    "discharge physical exam", "review of systems",
    "exam", "discharge exam", "admission exam",
    "discharge medications", "admission medications", "medications on admission",
    "medications on discharge", "medications", "home medications",
    "dexamethasone taper",
    "past medical history", "past surgical history", "surgical history",
    "social history", "family history", "obstetric history",
    "allergies", "drug allergies", "allergy",
    "pertinent results", "labs", "admission labs", "discharge labs",
    "laboratory data", "laboratory results",
    "laboratory findings", "incidental findings",
    "microbiology", "cultures", "blood cultures",
    "major surgical or invasive procedure", "procedures",
    "major surgical or invasive procedures", "procedure",
    "discharge instructions", "followup instructions", "follow-up instructions",
    "discharge condition", "discharge disposition",
    "code status", "advanced directives",
    "service", "attending", "date of birth", "sex", "unit",
    "admission date", "discharge date", "contact information",
    "assessment and plan", "plan", "assessment",
    "vital signs", "nutrition", "diet", "consults", "consultations",
    "what to expect when you go home", "what to expect at home",
    "medication changes", "colonoscopy findings", "endoscopy findings",
}


def load_auto_radiology_headers(path):
    section_list = pd.read_csv(path)
    headers = set()
    for category in RADIOLOGY_CATEGORIES:
        labels = section_list[section_list["category"].str.strip() == category]["label"].dropna()
        headers.update(l.strip().rstrip(":").strip().lower() for l in labels if l.strip())
    return headers


def load_normalization_dict(path):
    mapping = pd.read_csv(path)
    out = {}
    for _, row in mapping.iterrows():
        raw, canonical = str(row.iloc[0]).strip().lower(), str(row.iloc[1]).strip().lower()
        if raw and canonical:
            out[raw] = canonical
    return out


def build_canonical_radiology_set(normalization_dict):
    return {
        c for c in set(normalization_dict.values())
        if MODALITY_PATTERN.search(c) or SECTION_PATTERN.search(c)
    }


def make_classifier(auto_radiology_headers, normalization_dict, canonical_radiology_set):
    """Return classify_header(raw) -> (decision, reason).
    Priority: SAFE > category > keyword canonical > keyword fallback > KEEP."""

    def classify_header(raw_header):
        normalized = raw_header.strip().rstrip(":").strip().lower()
        if not normalized:
            return ("KEEP", "empty header")
        canonical = normalization_dict.get(normalized, normalized)

        if canonical in SAFE_CANONICAL_HEADERS or normalized in SAFE_CANONICAL_HEADERS:
            return ("KEEP", f"SAFE list (canonical: {canonical})")
        if normalized in auto_radiology_headers or canonical in auto_radiology_headers:
            return ("REMOVE", "Echo/ECG/Radiology category header")
        if canonical in canonical_radiology_set:
            return ("REMOVE", f"keyword-classified canonical: {canonical}")
        if normalized not in normalization_dict and FALLBACK_RADIOLOGY_PATTERN.search(normalized):
            return ("REMOVE", "keyword fallback (unmapped header)")
        return ("KEEP", f"default keep (canonical: {canonical})")

    return classify_header


def parse_sections(text, pattern):
    """Parse into [(header, content), ...]; the first header may be None (preamble)."""
    matches = list(pattern.finditer(text))
    if not matches:
        return [(None, text)]

    sections = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        if "\n" in header:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((header, text[match.end():end].strip()))
    return sections


def select_pattern(sample_texts):
    count_a = sum(len([h for h, _ in parse_sections(t, PATTERN_A) if h]) for t in sample_texts)
    count_b = sum(len([h for h, _ in parse_sections(t, PATTERN_B) if h]) for t in sample_texts)
    return PATTERN_A if count_a >= count_b else PATTERN_B


def process_notes(df, text_col, classify_header, pattern):
    cleaned_records, removal_records = [], []
    for idx in range(len(df)):
        row = df.iloc[idx]
        text = row[text_col]
        note_id = row.get("note_id", idx)

        kept = []
        for header, content in parse_sections(text, pattern):
            if header is None:
                kept.append((header, content))
                continue
            decision, reason = classify_header(header)
            if decision == "REMOVE":
                removal_records.append({
                    "note_id": note_id, "removed_header": header,
                    "removed_text_length": len(content), "reason": reason,
                })
            else:
                kept.append((header, content))

        cleaned_text = "\n\n".join(f"{h}:\n{c}" if h else c for h, c in kept)
        cleaned_records.append({
            "note_id": note_id, "subject_id": row.get("subject_id"),
            "hadm_id": row.get("hadm_id"), "cleaned_text": cleaned_text,
            "original_length": len(text), "cleaned_length": len(cleaned_text),
        })
        if (idx + 1) % 5000 == 0:
            print(f"Processed {idx + 1}/{len(df)} notes...")
    return cleaned_records, removal_records


def resolve_text_column(df):
    for c in ["text", "TEXT", "note_text", "discharge_text", "content", "cleaned_text"]:
        if c in df.columns:
            return c
    raise ValueError("No recognizable text column found.")


def main():
    auto_headers = load_auto_radiology_headers(SECTION_LIST_PATH)
    normalization_dict = load_normalization_dict(HEADER_MAPPING_PATH)
    canonical_radiology_set = build_canonical_radiology_set(normalization_dict)
    classify_header = make_classifier(auto_headers, normalization_dict, canonical_radiology_set)

    df = pd.read_csv(INPUT_NOTES_PATH)
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    text_col = resolve_text_column(df)
    print(f"Loaded {len(df):,} notes (text column: '{text_col}').")

    pattern = select_pattern(df[text_col].head(min(100, len(df))).tolist())
    cleaned_records, removal_records = process_notes(df, text_col, classify_header, pattern)

    cleaned_df = pd.DataFrame(cleaned_records)
    removal_df = pd.DataFrame(removal_records)
    print(f"Done. {len(cleaned_df):,} notes processed, {len(removal_df):,} sections removed.")

    cleaned_df[["note_id", "subject_id", "hadm_id", "cleaned_text"]].to_csv(
        OUTPUT_CLEANED_PATH, index=False, quoting=csv.QUOTE_ALL
    )
    if len(removal_df) > 0:
        removal_df.to_csv(OUTPUT_LOG_PATH, index=False)


if __name__ == "__main__":
    main()