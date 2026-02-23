import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import ta

import yfinance as yf

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# App config
# =========================
st.set_page_config(
    page_title="Tadawul Stock Decision Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Helpers
# =========================
TADAWUL_SUGGESTIONS = [
    ("2222.SR", "Saudi Aramco"),
    ("2010.SR", "SABIC"),
    ("1120.SR", "Al Rajhi Bank"),
    ("7010.SR", "STC"),
    ("1211.SR", "Ma'aden"),
    ("1180.SR", "SNB"),
    ("2380.SR", "Petro Rabigh"),
    ("^TASI.SR", "TASI Index"),
]


def normalize_tadawul_query(q: str) -> str:
    q = (q or "").strip().upper()

    name_map = {
        "ARAMCO": "2222.SR",
        "SAUDI ARAMCO": "2222.SR",
        "SABIC": "2010.SR",
        "STC": "7010.SR",
        "TASI": "^TASI.SR",
        "TADAWUL": "^TASI.SR",
    }
    if q in name_map:
        return name_map[q]

    # 4-digit -> Tadawul ticker
    if q.isdigit() and len(q) == 4:
        return f"{q}.SR"

    return q


def safe_get_metric(report, label, metric):
    return report.get(label, {}).get(metric, 0.0)


# =========================
# Data fetch (robust)
# =========================
@st.cache_data(ttl=60 * 30)
def fetch_ohlcv_yahoo_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Use yf.Ticker().history() instead of yf.download() to avoid multi-index issues.
    Returns columns: Date, Open, High, Low, Close, Volume, Change
    """
    t = yf.Ticker(ticker)

    # history end is exclusive; adding 1 day makes UI end-date feel inclusive
    end_plus = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = t.history(start=start, end=end_plus, interval="1d", auto_adjust=False, actions=False)

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. "
            "Try Tadawul format ####.SR (e.g., 1120.SR) or index ^TASI.SR."
        )

    # Ensure standard columns exist
    needed = {"Open", "High", "Low", "Close", "Volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Yahoo returned missing columns: {missing}. Returned columns: {list(df.columns)}")

    df = df.reset_index()

    # Sometimes index name is "Date", sometimes "Datetime"
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        else:
            # Try to find the first datetime-like column
            df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["Volume"] = df["Volume"].fillna(0)
    df["Change"] = df["Close"].pct_change() * 100

    return df


# =========================
# Indicators + rule signals for chart
# =========================
@st.cache_data
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["EMA20"] = ta.trend.ema_indicator(d["Close"], window=20)
    d["EMA50"] = ta.trend.ema_indicator(d["Close"], window=50)
    d["RSI14"] = ta.momentum.rsi(d["Close"], window=14)

    macd = ta.trend.MACD(d["Close"])
    d["MACD"] = macd.macd()
    d["MACD_signal"] = macd.macd_signal()
    d["MACD_diff"] = macd.macd_diff()

    d["ATR14"] = ta.volatility.average_true_range(d["High"], d["Low"], d["Close"], window=14)

    # simple rule markers (visual only)
    d["EMA20_above"] = d["EMA20"] > d["EMA50"]
    d["RuleBuy"] = d["EMA20_above"] & (~d["EMA20_above"].shift(1).fillna(False)) & (d["RSI14"] < 70)
    d["RuleSell"] = (~d["EMA20_above"]) & (d["EMA20_above"].shift(1).fillna(False)) | (d["RSI14"] > 70)

    return d


# =========================
# CART dataset (next-day target, leakage-free)
# =========================
@st.cache_data
def build_cart_dataset(df: pd.DataFrame):
    d = df.copy().sort_values("Date").reset_index(drop=True)

    d["ret_1d"] = d["Close"].pct_change() * 100
    d["ret_5d"] = d["Close"].pct_change(5) * 100
    d["vol_10d"] = d["ret_1d"].rolling(10).std()

    d["ema_10"] = ta.trend.ema_indicator(d["Close"], window=10)
    d["ema_20"] = ta.trend.ema_indicator(d["Close"], window=20)
    d["ema_50"] = ta.trend.ema_indicator(d["Close"], window=50)
    d["rsi_14"] = ta.momentum.rsi(d["Close"], window=14)

    macd = ta.trend.MACD(d["Close"])
    d["macd"] = macd.macd()
    d["macd_signal"] = macd.macd_signal()
    d["macd_diff"] = macd.macd_diff()

    d["atr_14"] = ta.volatility.average_true_range(d["High"], d["Low"], d["Close"], window=14)
    d["volume"] = pd.to_numeric(d.get("Volume", 0), errors="coerce").fillna(0)

    # next day return (target)
    d["next_ret_1d"] = d["Close"].shift(-1).pct_change() * 100

    base = float(d["ret_1d"].std()) if d["ret_1d"].notna().sum() > 30 else 1.0
    thr = max(0.5, 0.75 * base)

    def label(x):
        if pd.isna(x):
            return np.nan
        if x >= thr:
            return "Buy"
        if x <= -thr:
            return "Sell"
        return "Hold"

    d["Target"] = d["next_ret_1d"].apply(label)

    feature_cols = [
        "ret_1d", "ret_5d", "vol_10d",
        "ema_10", "ema_20", "ema_50",
        "rsi_14",
        "macd", "macd_signal", "macd_diff",
        "atr_14",
        "volume",
    ]

    # leakage prevention: use yesterday's indicators to predict tomorrow
    for c in feature_cols:
        d[c] = d[c].shift(1)

    d = d.dropna(subset=feature_cols + ["Target"]).reset_index(drop=True)
    return d, feature_cols, thr


def train_cart_time_split(dataset: pd.DataFrame, feature_cols: list, max_depth: int):
    X = dataset[feature_cols]
    y = dataset["Target"]

    split = int(len(dataset) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = DecisionTreeClassifier(criterion="gini", max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = model.classes_

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_test, y_pred, labels=classes),
        "classes": classes,
        "test_dates": dataset.iloc[split:]["Date"].reset_index(drop=True),
    }
    return model, metrics


# =========================
# Chart
# =========================
def candlestick_chart(df: pd.DataFrame, show_signals=True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC"
    ))

    if "EMA20" in df.columns and "EMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], mode="lines", name="EMA20"))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], mode="lines", name="EMA50"))

    if show_signals and "RuleBuy" in df.columns and "RuleSell" in df.columns:
        buys = df[df["RuleBuy"]]
        sells = df[df["RuleSell"]]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["Date"], y=buys["Close"], mode="markers",
                name="Rule Buy", marker=dict(symbol="triangle-up", size=10)
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["Date"], y=sells["Close"], mode="markers",
                name="Rule Sell", marker=dict(symbol="triangle-down", size=10)
            ))

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return fig


# ============================================================
# UI
# ============================================================
st.title("Tadawul Stock Decision Dashboard")
st.caption("Type a ticker/name → set start/end date → chart + improved CART next-day recommendation.")

# Sidebar
st.sidebar.header("Search & Date Range")

with st.sidebar.expander("Quick suggestions", expanded=True):
    for sym, name in TADAWUL_SUGGESTIONS:
        st.write(f"- **{sym}** — {name}")

query = st.sidebar.text_input(
    "Search (e.g., 2222, 2222.SR, ARAMCO, SABIC, ^TASI.SR)",
    value="2222"
)
ticker = normalize_tadawul_query(query)
st.sidebar.caption(f"Using ticker: **{ticker}**")

colA, colB = st.sidebar.columns(2)
with colA:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01").date())
with colB:
    end_date = st.date_input("End Date", value=pd.Timestamp.today().date())

if pd.Timestamp(end_date) <= pd.Timestamp(start_date):
    st.sidebar.error("End Date must be after Start Date.")
    st.stop()

show_rule_signals = st.sidebar.checkbox("Show rule signals on chart", value=True)

st.sidebar.header("CART Settings")
max_depth = st.sidebar.slider("Tree depth", 2, 12, 6)
conf_thresh = st.sidebar.slider("Confidence threshold", 0.34, 0.90, 0.55, step=0.01)

# Load data
try:
    raw = fetch_ohlcv_yahoo_history(ticker, str(start_date), str(end_date))
except Exception as e:
    st.error(f"Fetch failed for **{ticker}**.\n\n**Error:** {e}")
    st.info("Try: 2222.SR (Aramco), 2010.SR (SABIC), 1120.SR (Al Rajhi), ^TASI.SR (Index).")
    st.stop()

data = add_indicators(raw)

# KPIs
first_close = float(data.iloc[0]["Close"])
last_close = float(data.iloc[-1]["Close"])
period_return = (last_close / first_close - 1) * 100
volatility = float(data["Change"].std()) if data["Change"].notna().sum() > 5 else 0.0
running_max = data["Close"].cummax()
drawdown = (data["Close"] / running_max - 1) * 100
max_dd = float(drawdown.min()) if len(drawdown) else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Start Close", f"{first_close:.2f}")
k2.metric("End Close", f"{last_close:.2f}", f"{period_return:.2f}%")
k3.metric("Volatility (Std %)", f"{volatility:.2f}")
k4.metric("Max Drawdown", f"{max_dd:.2f}%")

# Main layout
left, right = st.columns([1.6, 1.0], gap="large")

with left:
    st.subheader("Price Chart (Candlestick + EMA20/EMA50)")
    st.plotly_chart(candlestick_chart(data, show_signals=show_rule_signals), use_container_width=True)

    st.subheader("Data Preview")
    st.dataframe(
        data[["Date", "Open", "High", "Low", "Close", "Volume", "Change", "EMA20", "EMA50", "RSI14"]],
        use_container_width=True
    )

with right:
    st.subheader("CART Next-Day Recommendation")

    dataset, feature_cols, thr = build_cart_dataset(data)

    if len(dataset) < 60:
        st.warning("Not enough rows after feature engineering. Increase date range.")
    else:
        model, m = train_cart_time_split(dataset, feature_cols, max_depth=max_depth)
        st.caption(
            f"Target = next-day return class using volatility threshold ±{thr:.2f}%. "
            "Features are shifted by 1 day (no leakage)."
        )
        st.metric("Holdout Accuracy (last 20% of period)", f"{m['accuracy']:.2f}")

        st.markdown("### Decision date (predict next day)")
        available_dates = dataset["Date"].dt.date.unique()
        chosen_date = st.selectbox("Pick a date", options=available_dates, index=max(0, len(available_dates) - 5))

        row = dataset[dataset["Date"].dt.date == chosen_date].iloc[-1]
        X_row = row[feature_cols].to_frame().T

        proba = model.predict_proba(X_row)[0]
        classes = model.classes_
        pred = classes[int(np.argmax(proba))]
        confidence = float(np.max(proba))

        if confidence >= conf_thresh:
            st.success(f"Recommendation: **{pred}** (confidence {confidence:.2f})")
        else:
            st.warning(f"Low confidence ({confidence:.2f}) → **Hold / No Trade**")

        st.markdown("### Example (based on your selected range)")
        st.write(
            f"On **{pd.Timestamp(chosen_date).date()}**, CART recommends **{pred}** for the **next trading day** "
            f"with confidence **{confidence:.2f}**."
        )
        st.write(f"Actual next-day return in data: **{row['next_ret_1d']:.2f}%** → label: **{row['Target']}**")

        st.markdown("### Confusion Matrix (test)")
        cm_df = pd.DataFrame(m["confusion"], index=m["classes"], columns=m["classes"])
        st.dataframe(cm_df, use_container_width=True)

        with st.expander("Decision Tree (Explainability)"):
            fig, ax = plt.subplots(figsize=(18, 8))
            plot_tree(model, feature_names=feature_cols, class_names=classes, filled=True, ax=ax)
            st.pyplot(fig)

st.markdown("---")
st.caption("Data source: Yahoo Finance (via yfinance). Tadawul tickers typically use ####.SR (e.g., 2222.SR).")