import logging
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


@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """Load multilingual embedding model once."""
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner=False)
def load_cpv_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"code", "description"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain columns: code, description")

    df = df[["code", "description"]].dropna().reset_index(drop=True)
    df["description"] = df["description"].astype(str)
    df["code"] = df["code"].astype(str)
    return df


@st.cache_resource(show_spinner=False)
def build_cpv_embeddings(descriptions: List[str]) -> np.ndarray:
    model = load_model()
    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings


def find_top_matches(
    tender_text: str,
    cpv_df: pd.DataFrame,
    cpv_embeddings: np.ndarray,
    top_k: int = 3,
) -> List[MatchResult]:
    model = load_model()
    query_embedding = model.encode(
        [tender_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = cpv_embeddings @ query_embedding
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = [
        MatchResult(
            code=cpv_df.iloc[idx]["code"],
            description=cpv_df.iloc[idx]["description"],
            similarity=float(scores[idx]),
        )
        for idx in top_indices
    ]
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

    tender_text = st.text_area(
        "Tender Text",
        placeholder="Paste tender text here...",
        height=220,
    )

    if st.button("Find Matching CPV Codes", type="primary"):
        if not tender_text.strip():
            st.warning("Please paste tender text first.")
            return

        logger.info("Received tender query of %s characters", len(tender_text))

        with st.spinner("Finding best CPV matches..."):
            matches = find_top_matches(tender_text, cpv_df, cpv_embeddings, top_k=3)

        st.subheader("Top Matches")
        for rank, match in enumerate(matches, start=1):
            st.markdown(
                f"**{rank}. {match.code} – {match.description}**  \n"
                f"Similarity: `{match.similarity:.2f}`"
            )

        with st.expander("Why these matches?"):
            st.write(
                "Matches are selected by cosine similarity between the tender embedding and "
                "CPV description embeddings. Higher score means stronger semantic alignment."
            )


if __name__ == "__main__":
    main()
