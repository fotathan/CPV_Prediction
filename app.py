import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tenderalign")


@dataclass
class MatchResult:
    code: str
    description: str
    similarity: float
    explanation: str


@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner=False)
def load_cpv_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"code", "description"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns: code, description")

    df = df[["code", "description"]].dropna().reset_index(drop=True)
    df["code"] = df["code"].astype(str)
    df["description"] = df["description"].astype(str)
    return df


@st.cache_resource(show_spinner=False)
def build_cpv_embeddings(descriptions: List[str]) -> np.ndarray:
    model = load_model()
    return model.encode(descriptions, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def extract_raw_cpv_mentions(tender_text: str) -> set[str]:
    canonical_mentions = set(re.findall(r"\b\d{8}-\d\b", tender_text))
    compact_mentions = {f"{c[:8]}-{c[8]}" for c in re.findall(r"\b\d{9}\b", tender_text)}
    return canonical_mentions | compact_mentions


def extract_cpv_stems(tender_text: str) -> set[str]:
    return set(re.findall(r"\b\d{8}\b", tender_text))


def extract_mentioned_cpv_codes(tender_text: str, cpv_df: pd.DataFrame) -> List[str]:
    known_codes = set(cpv_df["code"].tolist())
    stem_to_code = {code.split("-")[0]: code for code in known_codes}

    explicit = extract_raw_cpv_mentions(tender_text)
    explicit.update({stem_to_code[s] for s in extract_cpv_stems(tender_text) if s in stem_to_code})

    return sorted(code for code in explicit if code in known_codes)


def extract_text_cpv_candidates(tender_text: str, cpv_df: pd.DataFrame) -> List[tuple[str, str, str]]:
    """
    Return CPV-like mentions from tender text (########-# or ########) that are
    considered valid when their 4-digit parent group exists in the CPV list.
    """
    known_codes = set(cpv_df["code"].tolist())
    parent_map = {
        code.split("-")[0]: code
        for code in known_codes
        if code.split("-")[0].endswith("0000")
    }
    ordered_candidates: List[str] = []

    for match in re.finditer(r"\b\d{8}-\d\b|\b\d{8}\b", tender_text):
        candidate = match.group(0)
        if candidate not in ordered_candidates:
            ordered_candidates.append(candidate)

    validated: List[tuple[str, str, str]] = []
    for candidate in ordered_candidates:
        stem8 = candidate[:8]
        parent_code = parent_map.get(f"{stem8[:4]}0000")
        if not parent_code:
            continue

        if candidate in known_codes:
            description = cpv_df.loc[cpv_df["code"] == candidate, "description"].iloc[0]
        else:
            parent_desc = cpv_df.loc[cpv_df["code"] == parent_code, "description"].iloc[0]
            description = f"Text-mentioned CPV; parent group: {parent_code} ({parent_desc})"

        explanation = f"Found in tender text and validated via 4-digit parent group {parent_code}."
        validated.append((candidate, description, explanation))

    return validated


def find_top_matches(
    tender_text: str,
    cpv_df: pd.DataFrame,
    cpv_embeddings: np.ndarray,
    top_k: int = 3,
) -> List[MatchResult]:
    model = load_model()
    query_embedding = model.encode([tender_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]

    scores = cpv_embeddings @ query_embedding
    sorted_indices = np.argsort(scores)[::-1]

    validated_candidates = extract_text_cpv_candidates(tender_text, cpv_df)
    used_stems = set()
    results: List[MatchResult] = []

    for code, description, explanation in validated_candidates:
        stem = code[:8]
        if stem in used_stems:
            continue
        results.append(
            MatchResult(
                code=code,
                description=description,
                similarity=0.99,
                explanation=explanation,
            )
        )
        used_stems.add(stem)
        if len(results) == top_k:
            return results

    for idx in sorted_indices:
        code = cpv_df.iloc[idx]["code"]
        stem = code.split("-")[0]
        if stem in used_stems:
            continue
        results.append(
            MatchResult(
                code=code,
                description=cpv_df.iloc[idx]["description"],
                similarity=float(scores[idx]),
                explanation="Suggested by multilingual semantic similarity.",
            )
        )
        used_stems.add(stem)
        if len(results) == top_k:
            break

    return results


def main() -> None:
    st.set_page_config(page_title="TenderAlign AI", page_icon="🧭", layout="centered")
    st.title("TenderAlign AI")
    st.caption("Find the top 3 CPV codes from tender text using multilingual semantic similarity.")

    csv_path = Path("data/cpv_sample.csv")
    if not csv_path.exists():
        st.error("CPV CSV not found at data/cpv_sample.csv")
        return

    with st.spinner("Loading CPV data and embedding model..."):
        cpv_df = load_cpv_data(str(csv_path))
        cpv_embeddings = build_cpv_embeddings(cpv_df["description"].tolist())

    tender_text = st.text_area("Tender Text", placeholder="Paste tender text here...", height=220)

    if st.button("Find Matching CPV Codes", type="primary"):
        if not tender_text.strip():
            st.warning("Please paste tender text first.")
            return

        logger.info("Received tender query of %s characters", len(tender_text))

        raw_mentions = extract_raw_cpv_mentions(tender_text)
        raw_stems = extract_cpv_stems(tender_text)
        known_codes = set(cpv_df["code"].tolist())
        known_parent_stems = {
            code.split("-")[0]
            for code in known_codes
            if code.split("-")[0].endswith("0000")
        }
        missing_mentions = sorted(
            code for code in raw_mentions if f"{code[:4]}0000" not in known_parent_stems
        )
        missing_mentions.extend(
            stem for stem in sorted(raw_stems) if f"{stem[:4]}0000" not in known_parent_stems
        )

        with st.spinner("Finding best CPV matches..."):
            matches = find_top_matches(tender_text, cpv_df, cpv_embeddings, top_k=3)

        if missing_mentions:
            st.info(
                "Detected CPV-like mentions not found in the loaded CSV: "
                + ", ".join(missing_mentions[:5])
                + (" ..." if len(missing_mentions) > 5 else "")
                + ". Consider loading the full CPV list."
            )

        st.subheader("Top Matches")
        for rank, match in enumerate(matches, start=1):
            st.markdown(
                f"**{rank}. {match.code} – {match.description}**  \n"
                f"Similarity: `{match.similarity:.2f}`  \n"
                f"Why: {match.explanation}"
            )


if __name__ == "__main__":
    main()
