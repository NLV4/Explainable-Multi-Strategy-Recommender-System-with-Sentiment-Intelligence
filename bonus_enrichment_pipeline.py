"""Generate bonus artifacts for Cognitus Lite.

Creates:
- processed_items_for_streamlit_bonus.csv with safer FinBERT sentiment columns
- semantic_embeddings.npy with sentence-transformer embeddings

Run:
    python bonus_enrichment_pipeline.py
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


FINANCE_KEYWORDS = [
    "stock", "stocks", "market", "markets", "share", "shares", "investor", "investors",
    "inflation", "oil", "bank", "banks", "economy", "economic", "fed", "federal reserve",
    "interest rate", "interest rates", "bond", "bonds", "revenue", "profit", "profits",
    "loss", "losses", "earnings", "trade", "trading", "finance", "financial", "gdp",
    "currency", "currencies", "forex", "exchange rate", "crude", "nasdaq", "dow", "s&p",
    "sp500", "ipo", "merger", "acquisition", "dividend", "dividends", "treasury", "loan",
    "loans", "credit", "debt", "valuation", "capital", "fund", "funds", "etf", "commodity",
    "commodities", "price target", "guidance", "quarterly results"
]


def is_finance_text(text: str) -> bool:
    text = str(text).lower()
    return any(keyword in text for keyword in FINANCE_KEYWORDS)


def clean_finbert_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]


def label_to_score(label: str, confidence: float) -> float:
    label = str(label).lower()
    if label == "positive":
        return float(confidence)
    if label == "negative":
        return float(-confidence)
    return 0.0


def apply_safe_finbert(items: pd.DataFrame, min_confidence: float = 0.60) -> pd.DataFrame:
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
        batch_size=16,
    )

    texts = items["raw_text"].fillna("").astype(str).tolist()

    finance_mask = [is_finance_text(text) for text in texts]
    finance_indices = [i for i, is_fin in enumerate(finance_mask) if is_fin]
    finance_texts = [clean_finbert_text(texts[i]) for i in finance_indices]

    labels = ["neutral"] * len(items)
    confidences = [0.0] * len(items)
    signed_scores = [0.0] * len(items)
    backends = ["rule-neutral"] * len(items)

    if finance_texts:
        preds = classifier(finance_texts)

        for idx, pred in zip(finance_indices, preds):
            label = str(pred["label"]).lower()
            confidence = float(pred["score"])

            if confidence < min_confidence:
                labels[idx] = "neutral"
                confidences[idx] = confidence
                signed_scores[idx] = 0.0
                backends[idx] = "finbert-lowconf-neutral"
            else:
                labels[idx] = label
                confidences[idx] = confidence
                signed_scores[idx] = label_to_score(label, confidence)
                backends[idx] = "finbert"

    items["sentiment_label_finbert"] = labels
    items["sentiment_confidence_finbert"] = confidences
    items["sentiment_score_finbert"] = signed_scores
    items["sentiment_backend"] = backends

    # Make these the active sentiment columns used by the bonus app
    items["sentiment_label_active"] = labels
    items["sentiment_confidence_active"] = confidences
    items["sentiment_score_active"] = signed_scores

    print("Safe FinBERT sentiment created successfully.")
    print(f"Finance-relevant rows sent to FinBERT: {sum(finance_mask)} / {len(finance_mask)}")
    print(f"Rows forced to neutral by rule filter: {len(finance_mask) - sum(finance_mask)}")

    return items


def build_semantic_embeddings(items: pd.DataFrame, out_npy: Path) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = items["clean_text"].fillna("").astype(str).tolist()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(out_npy, embeddings)
    print(f"Saved semantic embeddings to {out_npy}")


def main() -> None:
    base = Path(__file__).resolve().parent
    csv_path = base / "processed_items_for_streamlit.csv"
    out_csv = base / "processed_items_for_streamlit_bonus.csv"
    out_npy = base / "semantic_embeddings.npy"

    items = pd.read_csv(csv_path)

    try:
        items = apply_safe_finbert(items, min_confidence=0.60)
    except Exception as exc:
        print(f"FinBERT step skipped: {exc}")

    items.to_csv(out_csv, index=False)
    print(f"Saved enriched CSV to {out_csv}")

    try:
        build_semantic_embeddings(items, out_npy)
    except Exception as exc:
        print(f"Embedding step skipped: {exc}")


if __name__ == "__main__":
    main()