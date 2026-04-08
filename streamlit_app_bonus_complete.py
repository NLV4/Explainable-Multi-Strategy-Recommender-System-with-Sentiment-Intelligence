from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

st.set_page_config(
    page_title="Cognitus Lite | Bonus Complete Finance Recommender",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

plotly_config = {
    "displayModeBar": True,
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
}

st.markdown("""
<style>
header[data-testid="stHeader"] {
    display: none !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
button[kind="header"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

CUSTOM_CSS = """
<style>
    :root {
        --bg: #07111f;
        --stroke: rgba(255, 255, 255, 0.12);
        --text: #f8fbff;
        --muted: #d3def2;
        --accent: #6ea8ff;
        --accent-2: #2dd4bf;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(91, 140, 255, 0.16), transparent 30%),
            radial-gradient(circle at top left, rgba(35, 198, 168, 0.12), transparent 30%),
            linear-gradient(180deg, #07111f 0%, #08101a 100%);
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1460px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 15, 27, 0.99), rgba(11, 18, 31, 0.99));
        border-right: 1px solid var(--stroke);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: var(--text);
    }

    .hero {
        background: linear-gradient(135deg, rgba(16, 30, 52, 0.96), rgba(12, 23, 38, 0.92));
        border: 1px solid var(--stroke);
        border-radius: 24px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.22);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(12, 22, 36, 0.96), rgba(9, 17, 28, 0.96));
        border: 1px solid var(--stroke);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        min-height: 110px;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.14);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.02em;
    }

    .glass-card {
        background: linear-gradient(180deg, rgba(11, 21, 35, 0.94), rgba(12, 20, 33, 0.90));
        border: 1px solid var(--stroke);
        border-radius: 22px;
        padding: 1rem 1.05rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.16);
        margin-bottom: 1rem;
    }

    .recommendation-card {
        background: linear-gradient(180deg, rgba(12, 22, 36, 0.96), rgba(8, 16, 28, 0.96));
        border: 1px solid var(--stroke);
        border-radius: 22px;
        padding: 1.1rem 1.1rem 0.9rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
    }

    .recommendation-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(91, 140, 255, 0.25), rgba(35, 198, 168, 0.22));
        border: 1px solid rgba(255,255,255,0.10);
        font-weight: 800;
        font-size: 0.9rem;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255, 255, 255, 0.05);
        font-size: 0.84rem;
        color: var(--text);
        margin-right: 0.4rem;
        margin-bottom: 0.45rem;
        font-weight: 600;
    }

    .pill-positive { background: rgba(35, 198, 168, 0.14); color: #8dffe1; border: 1px solid rgba(35,198,168,0.26); }
    .pill-neutral { background: rgba(245, 179, 66, 0.14); color: #ffe09a; border: 1px solid rgba(245,179,66,0.26); }
    .pill-negative { background: rgba(255, 107, 107, 0.14); color: #ffb0b0; border: 1px solid rgba(255,107,107,0.26); }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: var(--text) !important;
    }

    .stSelectbox, .stSlider, .stRadio, .stToggle {
        margin-bottom: 1rem !important;
    }

    [data-testid="stSidebar"] .stSelectbox > div,
    [data-testid="stSidebar"] .stSlider > div,
    [data-testid="stSidebar"] .stRadio > div {
        margin-top: 0.25rem !important;
    }

    div[data-baseweb="select"] > div {
        background: #0c1a2d !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        border-radius: 16px !important;
        min-height: 3.15rem !important;
        box-shadow: none !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span {
        color: #ffffff !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="select"] svg {
        fill: #ffffff !important;
    }

    div[data-baseweb="popover"] {
        z-index: 99999 !important;
    }

    div[data-baseweb="menu"] {
        background: #0c1a2d !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }

    div[data-baseweb="menu"] ul,
    div[data-baseweb="menu"] li,
    div[data-baseweb="menu"] div,
    [role="option"] {
        background: #0c1a2d !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="menu"] li:hover,
    div[data-baseweb="menu"] div:hover,
    [role="option"]:hover {
        background: #17304f !important;
        color: #ffffff !important;
    }

    [aria-selected="true"] {
        background: #203d63 !important;
        color: #ffffff !important;
    }

    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 0.35rem 0.65rem;
        margin-right: 0.35rem;
    }

    .stSlider label,
    .stRadio label,
    .stSelectbox label,
    .stToggle label {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px !important;
        min-height: 2.9rem !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        border: 0;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        box-shadow: 0 10px 24px rgba(91, 140, 255, 0.25);
    }

    .stDownloadButton > button {
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.05);
        color: var(--text);
    }

    button[data-baseweb="tab"] {
        color: #e8f0ff !important;
        font-weight: 700 !important;
    }

    .stCaption,
    [data-testid="stSidebar"] .stCaption {
        color: var(--muted) !important;
    }

    .pretty-table-outer {
        background: linear-gradient(180deg, rgba(11, 21, 35, 0.96), rgba(9, 17, 28, 0.96));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        box-shadow: 0 16px 36px rgba(0,0,0,0.18);
        margin-top: 0.5rem;
        overflow: hidden;
    }

    .pretty-table-scroll {
        overflow-x: auto;
        overflow-y: auto;
        max-height: 360px;
    }

    .pretty-table-scroll::-webkit-scrollbar {
        height: 10px;
        width: 10px;
    }

    .pretty-table-scroll::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.04);
        border-radius: 999px;
    }

    .pretty-table-scroll::-webkit-scrollbar-thumb {
        background: rgba(110,168,255,0.55);
        border-radius: 999px;
    }

    .pretty-table {
        width: 100%;
        min-width: 900px;
        border-collapse: collapse;
        font-size: 0.92rem;
    }

    .pretty-table thead th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #12243d;
        color: #f8fbff;
        text-align: left;
        padding: 0.85rem 0.9rem;
        font-weight: 800;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        white-space: nowrap;
    }

    .pretty-table tbody td {
        padding: 0.8rem 0.9rem;
        color: #eaf2ff;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        vertical-align: top;
        white-space: nowrap;
    }

    .pretty-table tbody tr:nth-child(even) {
        background: rgba(255,255,255,0.02);
    }

    .pretty-table tbody tr:hover {
        background: rgba(110,168,255,0.08);
    }

    .table-pill {
        display: inline-block;
        padding: 0.24rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.10);
    }

    .pill-pos {
        background: rgba(35, 198, 168, 0.14);
        color: #90ffe4;
    }

    .pill-neu {
        background: rgba(245, 179, 66, 0.14);
        color: #ffe09a;
    }

    .pill-neg {
        background: rgba(255, 107, 107, 0.14);
        color: #ffb0b0;
    }

    .headline-cell {
        min-width: 420px;
        max-width: 700px;
        white-space: normal !important;
        line-height: 1.45;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def safe_text(text, limit=220):
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def minmax_scale(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    max_v = np.nanmax(arr)
    min_v = np.nanmin(arr)
    if max_v - min_v == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_v) / (max_v - min_v)


def sentiment_label(score: float) -> str:
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


def tone_badge(label: str) -> str:
    css = {
        "positive": "pill-positive",
        "neutral": "pill-neutral",
        "negative": "pill-negative",
    }.get(label, "pill-neutral")
    return f"<span class='chip {css}'>{label.title()}</span>"


def render_pretty_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    df = df.head(max_rows).copy()

    html = """
    <div class="pretty-table-outer">
      <div class="pretty-table-scroll">
        <table class="pretty-table">
          <thead>
            <tr>
    """

    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            value = row[col]

            if col.lower() == "sentiment":
                label = str(value).lower()
                css = "pill-neu"
                if label == "positive":
                    css = "pill-pos"
                elif label == "negative":
                    css = "pill-neg"
                html += f"<td><span class='table-pill {css}'>{label.title()}</span></td>"
            elif "headline" in col.lower():
                html += f"<td class='headline-cell'>{value}</td>"
            else:
                html += f"<td>{value}</td>"
        html += "</tr>"

    html += """
          </tbody>
        </table>
      </div>
    </div>
    """
    return html


@st.cache_data
def load_data():
    base = Path(".")
    enriched_path = base / "processed_items_for_streamlit_bonus.csv"
    base_path = base / "processed_items_for_streamlit.csv"
    items = pd.read_csv(enriched_path if enriched_path.exists() else base_path)
    interactions = pd.read_csv(base / "train_interactions_for_streamlit.csv")
    users = pd.read_csv(base / "synthetic_user_profiles.csv")

    items["Date"] = pd.to_datetime(items["Date"], errors="coerce")
    interactions["Date"] = pd.to_datetime(interactions["Date"], errors="coerce")
    if "interaction" not in interactions.columns:
        interactions["interaction"] = 1

    if "sentiment_score_finbert" in items.columns:
        items["sentiment_score_active"] = items["sentiment_score_finbert"].fillna(items["sentiment_score"])
        items["sentiment_source"] = np.where(items["sentiment_score_finbert"].notna(), "FinBERT", "TextBlob")
    else:
        items["sentiment_score_active"] = items["sentiment_score"]
        items["sentiment_source"] = "TextBlob"

    items["sentiment_class"] = items["sentiment_score_active"].apply(sentiment_label)
    items["headline_preview"] = items["raw_text"].apply(lambda x: safe_text(x, 120))
    return items.sort_values("item_id").reset_index(drop=True), interactions, users


@st.cache_resource
def build_artifacts(items_df: pd.DataFrame, interactions_df: pd.DataFrame):
    tfidf = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        sublinear_tf=True,
    )
    content_matrix = tfidf.fit_transform(items_df["clean_text"].fillna(""))
    user_item_matrix = interactions_df.pivot_table(index="user_id", columns="item_id", values="interaction", fill_value=0)

    embedding_matrix = None
    emb_path = Path("semantic_embeddings.npy")
    if emb_path.exists():
        embedding_matrix = np.load(emb_path)
    return content_matrix, user_item_matrix, embedding_matrix


items_df, train_interactions, users_df = load_data()
content_matrix, user_item_matrix, embedding_matrix = build_artifacts(items_df, train_interactions)


def get_user_profile(user_id):
    row = users_df[users_df["user_id"] == user_id]
    if row.empty:
        return None
    return row.iloc[0]


def get_user_history(user_id: str, limit=8):
    hist = train_interactions[train_interactions["user_id"] == user_id].merge(
        items_df[["item_id", "Date", "market_mode", "risk_level", "sentiment_score_active", "raw_text"]],
        on="item_id",
        how="left",
        suffixes=("_interaction", "_item"),
    )

    date_candidates = ["Date", "Date_item", "Date_interaction", "Date_y", "Date_x"]
    chosen_date = None
    for col in date_candidates:
        if col in hist.columns:
            chosen_date = col
            break

    if chosen_date is not None:
        hist["Date"] = pd.to_datetime(hist[chosen_date], errors="coerce")
        hist = hist.sort_values("Date", ascending=False)

    hist["sentiment"] = hist["sentiment_score_active"].apply(sentiment_label)
    return hist.head(limit)


def dominant_signal_name(row):
    signal_map = {
        "Semantic" if "embedding_score" in row else "Content": row.get("embedding_score", row.get("content_score", 0)),
        "Collaborative": row.get("cf_score", 0),
        "Context": row.get("context_score", 0),
        "Sentiment": row.get("sentiment_component", 0),
    }
    return max(signal_map, key=signal_map.get)


def recommend_hybrid(
    user_id,
    current_market_mode="bullish",
    current_risk="medium",
    preferred_sentiment="positive",
    top_k=5,
    mode="Classic TF-IDF",
    w_content=0.30,
    w_cf=0.20,
    w_context=0.20,
    w_sentiment=0.10,
    w_embedding=0.20,
):
    seen_items = set(train_interactions.loc[train_interactions["user_id"] == user_id, "item_id"].tolist())

    content_scores = np.zeros(len(items_df))
    if len(seen_items) > 0:
        seen_list = sorted(list(seen_items))
        user_profile_vec = np.asarray(content_matrix[seen_list].mean(axis=0)).reshape(1, -1)
        content_scores = cosine_similarity(user_profile_vec, content_matrix).ravel()

    embedding_scores = np.zeros(len(items_df))
    use_embeddings = mode == "Bonus Semantic Mode" and embedding_matrix is not None and len(seen_items) > 0
    if use_embeddings:
        seen_list = sorted(list(seen_items))
        profile_vec = embedding_matrix[seen_list].mean(axis=0).reshape(1, -1)
        embedding_scores = cosine_similarity(profile_vec, embedding_matrix).ravel()

    cf_scores = np.zeros(len(items_df))
    if user_id in user_item_matrix.index:
        user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
        similarities = cosine_similarity(user_vector, user_item_matrix.values).ravel()
        similar_users = pd.DataFrame({"user_id": user_item_matrix.index, "similarity": similarities}).sort_values("similarity", ascending=False)
        for _, row in similar_users.iterrows():
            other_user = row["user_id"]
            sim = row["similarity"]
            if other_user == user_id:
                continue
            other_items = train_interactions.loc[train_interactions["user_id"] == other_user, "item_id"].tolist()
            for item in other_items:
                if item not in seen_items and item < len(cf_scores):
                    cf_scores[item] += sim

    context_scores = np.zeros(len(items_df))
    for i in range(len(items_df)):
        score = 0.0
        if items_df.loc[i, "market_mode"] == current_market_mode:
            score += 0.5
        if items_df.loc[i, "risk_level"] == current_risk:
            score += 0.3
        if sentiment_label(items_df.loc[i, "sentiment_score_active"]) == preferred_sentiment:
            score += 0.2
        context_scores[i] = score

    sentiment_scores = items_df["sentiment_score_active"].fillna(0).values.copy()
    if preferred_sentiment == "negative":
        sentiment_scores = -sentiment_scores
    elif preferred_sentiment == "neutral":
        sentiment_scores = -np.abs(sentiment_scores)

    content_scaled = minmax_scale(content_scores)
    embed_scaled = minmax_scale(embedding_scores)
    cf_scaled = minmax_scale(cf_scores)
    context_scaled = minmax_scale(context_scores)
    sentiment_scaled = minmax_scale(sentiment_scores)

    if use_embeddings:
        final_scores = (
            w_content * content_scaled +
            w_embedding * embed_scaled +
            w_cf * cf_scaled +
            w_context * context_scaled +
            w_sentiment * sentiment_scaled
        )
    else:
        final_scores = (
            (w_content + w_embedding) * content_scaled +
            w_cf * cf_scaled +
            w_context * context_scaled +
            w_sentiment * sentiment_scaled
        )

    candidate_idx = [i for i in items_df["item_id"].tolist() if i not in seen_items]
    ranked = sorted(candidate_idx, key=lambda i: final_scores[i], reverse=True)[:top_k]
    result = items_df[items_df["item_id"].isin(ranked)].copy()
    result["content_score"] = result["item_id"].apply(lambda i: float(content_scaled[i]))
    result["embedding_score"] = result["item_id"].apply(lambda i: float(embed_scaled[i]))
    result["cf_score"] = result["item_id"].apply(lambda i: float(cf_scaled[i]))
    result["context_score"] = result["item_id"].apply(lambda i: float(context_scaled[i]))
    result["sentiment_component"] = result["item_id"].apply(lambda i: float(sentiment_scaled[i]))
    result["hybrid_score"] = result["item_id"].apply(lambda i: float(final_scores[i]))

    def explain(row):
        reasons = []
        if use_embeddings and row["embedding_score"] > 0.30:
            reasons.append("semantic embedding similarity matched the user history")
        elif row["content_score"] > 0.30:
            reasons.append("headline pattern is close to the user history")
        if row["cf_score"] > 0.20:
            reasons.append("similar users also interacted with this market day")
        if row["market_mode"] == current_market_mode:
            reasons.append(f"matches the current market regime ({current_market_mode})")
        if row["risk_level"] == current_risk:
            reasons.append(f"aligns with the chosen risk setting ({current_risk})")
        reasons.append(f"news tone is {sentiment_label(row['sentiment_score_active'])}")
        reasons.append("final rank is produced by a weighted hybrid engine")
        return "; ".join(reasons)

    result["explanation"] = result.apply(explain, axis=1)
    return result.sort_values("hybrid_score", ascending=False).reset_index(drop=True)


st.sidebar.markdown("## 🧠 Investor Scenario")
st.sidebar.caption("Bonus-complete version with semantic retrieval, FinBERT-ready sentiment, and live refresh controls.")

if st_autorefresh is not None:
    auto_refresh = st.sidebar.toggle("Enable live refresh", value=False)
    refresh_seconds = st.sidebar.slider("Refresh every (seconds)", 5, 60, 15)
    if auto_refresh:
        st_autorefresh(interval=refresh_seconds * 1000, key="datarefresh")
else:
    st.sidebar.info("Install streamlit-autorefresh to enable live refresh mode.")

selected_user = st.sidebar.selectbox("Investor type", users_df["user_id"].tolist(), index=0)
profile = get_user_profile(selected_user)
default_market = profile["pref_market"] if profile is not None else "bullish"
default_risk = profile["pref_risk"] if profile is not None else "medium"
default_sentiment = profile["pref_sentiment"] if profile is not None else "positive"

market_options = sorted(items_df["market_mode"].dropna().unique().tolist())
risk_options = sorted(items_df["risk_level"].dropna().unique().tolist())
sentiment_options = ["positive", "neutral", "negative"]

current_market = st.sidebar.radio("Current market regime", market_options, index=market_options.index(default_market), horizontal=True)
current_risk = st.sidebar.select_slider("Risk environment", options=risk_options, value=default_risk)
preferred_sentiment = st.sidebar.selectbox("News tone focus", sentiment_options, index=sentiment_options.index(default_sentiment))
num_recs = st.sidebar.slider("How many ranked insights to review", 3, 10, 5)
mode = st.sidebar.radio("Recommendation mode", ["Classic TF-IDF", "Bonus Semantic Mode"], index=1 if embedding_matrix is not None else 0)

with st.sidebar.expander("⚙️ Hybrid weights"):
    w_content = st.slider("TF-IDF content", 0.0, 1.0, 0.30, 0.05)
    w_embedding = st.slider("Semantic embeddings", 0.0, 1.0, 0.20 if embedding_matrix is not None else 0.0, 0.05)
    w_cf = st.slider("Collaborative", 0.0, 1.0, 0.20, 0.05)
    w_context = st.slider("Context", 0.0, 1.0, 0.20, 0.05)
    w_sentiment = st.slider("Sentiment", 0.0, 1.0, 0.10, 0.05)
    total = w_content + w_embedding + w_cf + w_context + w_sentiment
    if total == 0:
        st.stop()
    w_content /= total
    w_embedding /= total
    w_cf /= total
    w_context /= total
    w_sentiment /= total

st.markdown("""
<div class='hero'>
    <div style='font-size:2rem; font-weight:800; line-height:1.1;'>Cognitus Lite+</div>
    <div style='font-size:1.05rem; font-weight:700; margin-bottom:0.2rem;'>Bonus-Complete Explainable Hybrid Finance Recommendation Dashboard</div>
    <div style='color:#aab8d3;'>This enhanced version adds FinBERT-ready sentiment support, semantic embedding search, live-refresh controls, deployment files, and the original hybrid explainable recommender.</div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
for col, (label, value) in zip(
    [k1, k2, k3, k4],
    [
        ("Items", f"{len(items_df):,}"),
        ("Users", f"{users_df['user_id'].nunique():,}"),
        ("Interactions", f"{len(train_interactions):,}"),
        ("Sentiment backend", items_df["sentiment_source"].iloc[0] if items_df["sentiment_source"].nunique() == 1 else "Mixed"),
    ],
):
    with col:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

left, right = st.columns([1.2, 1], gap="large")
with left:
    st.markdown("<div class='glass-card'><h3>User profile</h3>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Preferred market", profile["pref_market"].title())
    p2.metric("Preferred risk", profile["pref_risk"].title())
    p3.metric("Preferred sentiment", profile["pref_sentiment"].title())
    hist = get_user_history(selected_user)
    hist_view = hist[["Date", "item_id", "market_mode", "risk_level", "sentiment", "raw_text"]].copy()
    hist_view["Date"] = hist_view["Date"].dt.strftime("%Y-%m-%d")
    hist_view["headline_preview"] = hist_view["raw_text"].apply(lambda x: safe_text(x, 90))
    table_df = hist_view[["Date", "item_id", "market_mode", "risk_level", "sentiment", "headline_preview"]].copy()
    st.markdown(render_pretty_table(table_df, max_rows=8), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='glass-card'><h3>Scenario briefing</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<span class='chip'>Investor: {selected_user}</span>"
        f"<span class='chip'>Market: {current_market.title()}</span>"
        f"<span class='chip'>Risk: {current_risk.title()}</span>"
        f"<span class='chip'>Tone lens: {preferred_sentiment.title()}</span>"
        f"<span class='chip'>Mode: {mode}</span>",
        unsafe_allow_html=True,
    )
    sentiment_counts = items_df["sentiment_class"].value_counts().reindex(["negative", "neutral", "positive"]).fillna(0).reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig_sent = px.bar(sentiment_counts, x="sentiment", y="count", color="sentiment")
    fig_sent.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaf2ff"),
    )
    st.plotly_chart(fig_sent, use_container_width=True, config=plotly_config)
    st.markdown("</div>", unsafe_allow_html=True)

reco = recommend_hybrid(
    user_id=selected_user,
    current_market_mode=current_market,
    current_risk=current_risk,
    preferred_sentiment=preferred_sentiment,
    top_k=num_recs,
    mode=mode,
    w_content=w_content,
    w_cf=w_cf,
    w_context=w_context,
    w_sentiment=w_sentiment,
    w_embedding=w_embedding,
)

t1, t2, t3, t4 = st.tabs(["Recommendations", "Deep analysis", "Bonus features", "Deployment"])

with t1:
    for idx, row in reco.iterrows():
        st.markdown(
            f"""
            <div class='recommendation-card'>
                <div style='display:flex; align-items:center; justify-content:space-between;'>
                    <div style='display:flex; gap:0.75rem; align-items:center;'>
                        <div class='recommendation-rank'>#{idx + 1}</div>
                        <div><div style='font-weight:700;'>{pd.to_datetime(row['Date']).strftime('%Y-%m-%d')}</div><div style='color:#aab8d3;'>Driver: {dominant_signal_name(row)}</div></div>
                    </div>
                    <div style='font-size:1.35rem; font-weight:800;'>{row['hybrid_score']:.3f}</div>
                </div>
                <div style='margin-top:0.4rem;'>{tone_badge(sentiment_label(row['sentiment_score_active']))}<span class='chip'>Market: {row['market_mode']}</span><span class='chip'>Risk: {row['risk_level']}</span></div>
                <div style='margin-top:0.6rem; font-size:1rem; line-height:1.45;'>{safe_text(row['raw_text'], 240)}</div>
                <div style='margin-top:0.45rem; color:#aab8d3;'><strong>Why it surfaced:</strong> {row['explanation']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with t2:
    c1, c2 = st.columns(2)
    with c1:
        component_means = pd.DataFrame({
            "component": ["tfidf", "semantic", "collaborative", "context", "sentiment"],
            "score": [
                reco["content_score"].mean(),
                reco["embedding_score"].mean(),
                reco["cf_score"].mean(),
                reco["context_score"].mean(),
                reco["sentiment_component"].mean()
            ]
        })
        fig_avg = px.bar(component_means, x="component", y="score", color="component")
        fig_avg.update_layout(
            height=320,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eaf2ff"),
        )
        st.plotly_chart(fig_avg, use_container_width=True, config=plotly_config)
    with c2:
        deep_df = reco[["item_id", "Date", "content_score", "embedding_score", "cf_score", "context_score", "sentiment_component", "hybrid_score"]].copy()
        deep_df["Date"] = pd.to_datetime(deep_df["Date"]).dt.strftime("%Y-%m-%d")
        st.markdown(render_pretty_table(deep_df, max_rows=8), unsafe_allow_html=True)

with t3:
    st.markdown("""
    ### Bonus feature audit
    - **BERT sentiment:** enabled when `processed_items_for_streamlit_bonus.csv` contains `sentiment_score_finbert`.
    - **Embedding/vector search:** enabled when `semantic_embeddings.npy` is present.
    - **Real-time updates:** enabled with `streamlit-autorefresh`.
    - **Cloud deployment readiness:** deployment files are provided in this package.
    """)
    st.success("This app is ready to use the bonus features as soon as the enriched artifacts are generated from the updated notebook.")
    st.code("python -m streamlit run streamlit_app_bonus_complete.py")

with t4:
    st.markdown("""
    ### Deployment-ready files
    Use one of these options:
    1. **Streamlit Community Cloud / Render** with `requirements_streamlit_bonus.txt`
    2. **Docker** with the provided `Dockerfile`
    3. **Hugging Face Spaces** using Streamlit mode

    Note: actual cloud hosting must still be performed on the selected platform.
    """)
    st.code("docker build -t cognitus-lite .\\ndocker run -p 8501:8501 cognitus-lite")

st.markdown(
    "<div style='color:#aab8d3; margin-top:1rem;'>Built from processed_items_for_streamlit.csv, train_interactions_for_streamlit.csv, synthetic_user_profiles.csv, and optional enriched bonus artifacts.</div>",
    unsafe_allow_html=True,
)