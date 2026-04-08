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
    "stock", "stocks", "share", "shares",
    "market", "markets",
    "investor", "investors",
    "nasdaq", "dow", "s&p", "sp500",
    "ipo", "dividend", "dividends",
    "earnings", "revenue", "profit", "profits", "loss", "losses",
    "inflation", "interest rate", "interest rates",
    "bond", "bonds", "treasury",
    "forex", "currency", "currencies", "exchange rate",
    "bank", "banks", "banking",
    "oil", "crude",
    "gdp", "economy", "economic",
    "financial", "finance",
    "trade", "trading",
    "etf", "fund", "funds",
    "credit", "debt", "capital",
    "merger", "acquisition",
    "quarterly results", "guidance"
]


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9&\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_finance_text(text: str) -> bool:
    """
    Strict domain filter:
    only classify as finance if at least one strong finance phrase appears.
    """
    text = normalize_text(text)
    padded = f" {text} "

    for keyword in FINANCE_KEYWORDS:
        kw = normalize_text(keyword)
        if f" {kw} " in padded:
            return True
    return False


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


def apply_safe_finbert(items: pd.DataFrame, min_confidence: float = 0.65) -> pd.DataFrame:
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )

    texts = items["raw_text"].fillna("").astype(str).tolist()

    labels = []
    confidences = []
    signed_scores = []
    backends = []

    finance_count = 0
    neutral_rule_count = 0
    lowconf_count = 0

    for text in texts:
        cleaned = clean_finbert_text(text)

        # Rule 1: non-financial text -> neutral directly
        if not cleaned or not is_finance_text(cleaned):
            labels.append("neutral")
            confidences.append(0.0)
            signed_scores.append(0.0)
            backends.append("rule-neutral")
            neutral_rule_count += 1
            continue

        finance_count += 1
        pred = classifier(cleaned)[0]

        pred_label = str(pred["label"]).lower()
        confidence = float(pred["score"])

        # Rule 2: class-specific confidence thresholds
        if pred_label == "positive" and confidence < 0.40:
            labels.append("neutral")
            confidences.append(confidence)
            signed_scores.append(0.0)
            backends.append("finbert-lowconf-neutral")
            lowconf_count += 1
            continue

        if pred_label == "negative" and confidence < 0.60:
            labels.append("neutral")
            confidences.append(confidence)
            signed_scores.append(0.0)
            backends.append("finbert-lowconf-neutral")
            lowconf_count += 1
            continue

        # Rule 3: confident finance prediction -> use FinBERT output
        labels.append(pred_label)
        confidences.append(confidence)
        signed_scores.append(label_to_score(pred_label, confidence))
        backends.append("finbert")

    items["sentiment_label_finbert"] = labels
    items["sentiment_confidence_finbert"] = confidences
    items["sentiment_score_finbert"] = signed_scores
    items["sentiment_backend"] = backends

    # Active columns used by the bonus app
    items["sentiment_label_active"] = labels
    items["sentiment_confidence_active"] = confidences
    items["sentiment_score_active"] = signed_scores

    print("Safe FinBERT sentiment created successfully.")
    print(f"Finance-relevant rows sent to FinBERT: {finance_count}")
    print(f"Rows forced neutral by domain filter: {neutral_rule_count}")
    print(f"Finance rows forced neutral by low confidence: {lowconf_count}")

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

    items = apply_safe_finbert(items, min_confidence=0.55)
    items.to_csv(out_csv, index=False)
    print(f"Saved enriched CSV to {out_csv}")

    build_semantic_embeddings(items, out_npy)


if __name__ == "__main__":
    main()