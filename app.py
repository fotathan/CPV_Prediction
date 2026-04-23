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

DEFAULT_RESULT_LANGUAGE = "EN"
LANGUAGE_COLUMNS = [
    "BG",
    "CS",
    "DA",
    "DE",
    "EL",
    "EN",
    "ES",
    "ET",
    "FI",
    "FR",
    "GA",
    "HR",
    "HU",
    "IT",
    "LT",
    "LV",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SL",
    "SV",
]


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
    if "CODE" not in df.columns:
        raise ValueError("CSV must contain column: CODE")

    available_langs = [col for col in LANGUAGE_COLUMNS if col in df.columns]
    if DEFAULT_RESULT_LANGUAGE not in available_langs:
        raise ValueError("CSV must contain EN column for fallback descriptions")

    keep_columns = ["CODE"] + available_langs
    df = df[keep_columns].dropna(subset=["CODE"]).reset_index(drop=True)
    df["CODE"] = df["CODE"].astype(str)

    for col in available_langs:
        df[col] = df[col].astype(str)
        df.loc[df[col].str.strip().eq(""), col] = np.nan
        df.loc[df[col].str.lower().isin(["nan", "none"]), col] = np.nan

    df["embedding_text"] = df[DEFAULT_RESULT_LANGUAGE].fillna("")
    missing_en = df["embedding_text"].str.strip().eq("")
    for col in available_langs:
        if col == DEFAULT_RESULT_LANGUAGE:
            continue
        df.loc[missing_en, "embedding_text"] = df.loc[missing_en, col].fillna("")
        missing_en = df["embedding_text"].str.strip().eq("")
        if not missing_en.any():
            break

    df = df[df["embedding_text"].str.strip() != ""].reset_index(drop=True)
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


def get_display_description(row: pd.Series, selected_language: str) -> str:
    selected = row.get(selected_language)
    if pd.notna(selected) and str(selected).strip():
        return str(selected)
    english = row.get(DEFAULT_RESULT_LANGUAGE)
    if pd.notna(english) and str(english).strip():
        return str(english)
    return str(row.get("embedding_text", ""))


def extract_text_cpv_candidates(
    tender_text: str,
    cpv_df: pd.DataFrame,
    selected_language: str,
) -> List[tuple[str, str, str]]:
    """
    Return CPV-like mentions from tender text (########-# or ########) that are
    considered valid when their 4-digit parent group exists in the CPV list.
    """
    known_codes = set(cpv_df["CODE"].tolist())
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
            exact_row = cpv_df.loc[cpv_df["CODE"] == candidate].iloc[0]
            description = get_display_description(exact_row, selected_language)
        else:
            parent_row = cpv_df.loc[cpv_df["CODE"] == parent_code].iloc[0]
            parent_desc = get_display_description(parent_row, selected_language)
            description = f"Text-mentioned CPV; parent group: {parent_code} ({parent_desc})"

        explanation = f"Found in tender text and validated via 4-digit parent group {parent_code}."
        validated.append((candidate, description, explanation))

    return validated


def find_top_matches(
    tender_text: str,
    cpv_df: pd.DataFrame,
    cpv_embeddings: np.ndarray,
    selected_language: str,
    top_k: int = 3,
) -> List[MatchResult]:
    model = load_model()
    query_embedding = model.encode([tender_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]

    scores = cpv_embeddings @ query_embedding
    sorted_indices = np.argsort(scores)[::-1]

    validated_candidates = extract_text_cpv_candidates(tender_text, cpv_df, selected_language)
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
        code = cpv_df.iloc[idx]["CODE"]
        stem = code.split("-")[0]
        if stem in used_stems:
            continue
        results.append(
            MatchResult(
                code=code,
                description=get_display_description(cpv_df.iloc[idx], selected_language),
                similarity=float(scores[idx]),
                explanation="Suggested by multilingual semantic similarity.",
            )
        )
        used_stems.add(stem)
        if len(results) == top_k:
            break

    return results


def resolve_cpv_csv_path() -> Path:
    candidates = [
        Path("/data/cpv_full.csv"),
        Path("data/cpv_full.csv"),
        Path("data/cpv_sample.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("/data/cpv_full.csv")


def main() -> None:
    st.set_page_config(page_title="TenderAlign AI", page_icon="🧭", layout="centered")
    st.title("TenderAlign AI")
    st.caption("Find the top 3 CPV codes from tender text using multilingual semantic similarity.")

    csv_path = resolve_cpv_csv_path()
    if not csv_path.exists():
        st.error("CPV CSV not found. Expected /data/cpv_full.csv or data/cpv_full.csv")
        return

    with st.spinner("Loading CPV data and embedding model..."):
        cpv_df = load_cpv_data(str(csv_path))
        cpv_embeddings = build_cpv_embeddings(cpv_df["embedding_text"].tolist())

    available_languages = [col for col in LANGUAGE_COLUMNS if col in cpv_df.columns]
    selected_language = st.selectbox(
        "Result description language",
        options=available_languages,
        index=available_languages.index(DEFAULT_RESULT_LANGUAGE) if DEFAULT_RESULT_LANGUAGE in available_languages else 0,
    )

    st.caption(f"Loaded CPV file: {csv_path}")
    tender_text = st.text_area("Tender Text", placeholder="Paste tender text here...", height=220)

    if st.button("Find Matching CPV Codes", type="primary"):
        if not tender_text.strip():
            st.warning("Please paste tender text first.")
            return

        logger.info("Received tender query of %s characters", len(tender_text))

        raw_mentions = extract_raw_cpv_mentions(tender_text)
        raw_stems = extract_cpv_stems(tender_text)
        known_codes = set(cpv_df["CODE"].tolist())
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
            matches = find_top_matches(
                tender_text,
                cpv_df,
                cpv_embeddings,
                selected_language=selected_language,
                top_k=3,
            )

        if missing_mentions:
            st.info(
                "Detected CPV-like mentions not validated by available parent groups in the loaded CSV: "
                + ", ".join(missing_mentions[:5])
                + (" ..." if len(missing_mentions) > 5 else "")
                + "."
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
