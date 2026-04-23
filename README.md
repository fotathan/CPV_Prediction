# TenderAlign AI (MVP)

TenderAlign AI is a lightweight Streamlit app that maps tender text to the **top 3 most relevant CPV codes** using multilingual semantic embeddings and cosine similarity.

## Features

- Paste tender text in multiple languages (e.g., Greek, English, German)
- Returns top 3 CPV matches with:
  - CPV code
  - CPV description
  - Similarity score (0-1)
- If the tender text already contains valid CPV codes from the dataset, those codes are prioritized in results
- CPV detection supports `########-#`, compact `#########`, and 8-digit stems `########` (validated against known CPV dataset codes)
- In-memory CPV embeddings generated once at app startup
- Fast retrieval suitable for hundreds to thousands of CPV rows

## Project structure

- `app.py`: Streamlit UI + matching logic
- `data/cpv_sample.csv`: sample CPV data (`code`, `description`)
- `requirements.txt`: Python dependencies

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## CSV format

The app expects a CSV at `data/cpv_sample.csv` with columns:

- `code` (example: `15900000-7`)
- `description` (example: `Beverages tobacco and related products`)

You can replace the sample file with your full CPV list.
If codes mentioned in tenders are missing from your CSV, the app will notify you.

## How it works

1. Loads CPV rows from CSV
2. Uses `paraphrase-multilingual-MiniLM-L12-v2` to embed all CPV descriptions
3. Embeds the tender text at query time
4. Computes cosine similarity
5. Prioritizes CPV codes explicitly found in the tender text
6. Fills remaining slots by cosine-similarity ranking to return top 3 results

## Notes

- First run may take longer while model files are downloaded.
- For large inputs, consider trimming legal boilerplate before search for better relevance.
- If you hit an `IndentationError`, ensure your deployed `app.py` exactly matches the latest committed version (merge conflict markers or partial conflict resolutions can break Python indentation).
