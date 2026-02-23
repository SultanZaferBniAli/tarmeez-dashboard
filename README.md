
# Tadawul Stock Decision Dashboard (CART-Based Buy/Hold/Sell Recommendations)

## Project Overview

This project presents a live, interactive financial dashboard that helps non-technical stakeholders understand market behavior and make clearer, data-informed buy/hold/sell decisions for Saudi Exchange (Tadawul) equities. I chose Saudi stocks as the focus because Tadawul is the primary market for Saudi Arabia and a core venue for domestic investors, institutions, and fintech use cases. The central question the dashboard addresses is: **“Given recent price action and technical signals within a selected date range, what is the model-supported recommendation for the next trading day, and how confident is that recommendation?”**

To answer this, the dashboard integrates market data retrieval, technical indicators, and a CART (Classification and Regression Tree) model that predicts **next-day** direction as Buy, Hold, or Sell. The app emphasizes interpretability (decision trees, probability outputs) so stakeholders can understand *why* a recommendation is produced, not just the result. The application is built with Streamlit to ensure a simple, shareable, live interface suitable for business and product teams.
**References:**

* Saudi Exchange (Tadawul) overview: [https://www.saudiexchange.sa](https://www.saudiexchange.sa)
* Streamlit (interactive dashboards): [https://streamlit.io](https://streamlit.io)
* Decision Trees (CART) overview: [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)

## Data Source

The dashboard retrieves historical daily OHLCV (Open, High, Low, Close, Volume) data from **Yahoo Finance** via the open-source `yfinance` library. This source is widely used in research and prototyping for financial analytics because it provides consistent access to historical equity and index data, including Saudi Exchange tickers in the format `####.SR` (e.g., `2222.SR` for Saudi Aramco) and the Tadawul All Share Index (`^TASI.SR`). The data includes prices and volumes necessary for computing returns and technical indicators (e.g., EMA, RSI, MACD), which are standard inputs in quantitative analysis.
**References:**

* Yahoo Finance: [https://finance.yahoo.com](https://finance.yahoo.com)
* yfinance documentation: [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)
* Technical Analysis indicators reference: [https://technical-analysis-library-in-python.readthedocs.io/en/latest/](https://technical-analysis-library-in-python.readthedocs.io/en/latest/)

## Steps & Methodology

The workflow begins by fetching daily OHLCV data for a user-selected ticker and date range. The data is cleaned by enforcing numeric types, sorting by date, and removing incomplete rows. Feature engineering then derives returns and technical indicators (EMA, RSI, MACD, ATR), which are commonly used in financial analysis to capture trend, momentum, and volatility.

For modeling, I implemented a **CART classifier** using scikit-learn. To avoid look-ahead bias (data leakage), all features are shifted so that the model only uses information available *before* the prediction date. The target variable is defined as the **next-day return class** (Buy/Sell/Hold), using a volatility-adjusted threshold rather than an arbitrary fixed percentage. The dataset is split by time (first 80% for training, last 20% for testing) to better reflect real-world deployment where future data is unknown.

The dashboard visualizes candlestick charts with overlays (EMAs and optional rule-based signals) and reports model performance using accuracy, confusion matrices, and class-level precision/recall. The final recommendation is presented with **prediction probabilities** and a confidence threshold, so the system can explicitly advise “Hold / No Trade” when confidence is low. This design choice prioritizes responsible analytics and reduces over-trading risk.
**References:**

* scikit-learn (DecisionTreeClassifier): [https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* Data leakage in ML (best practices): [https://scikit-learn.org/stable/common_pitfalls.html#data-leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
* Time-aware model evaluation: [https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

---

## Key Insights

Across tested Tadawul equities, short-term price movements exhibit regime changes that are partially captured by momentum and trend indicators (e.g., EMA crossovers and RSI extremes). The CART model performs best in trending periods and struggles during sideways markets, which is expected for tree-based classifiers on noisy financial time series. The confidence threshold helps mitigate false positives by recommending “Hold” when signals are ambiguous. For stakeholders, the key takeaway is that **model-supported recommendations can augment human judgment**, especially for screening opportunities and understanding risk regimes, but should not replace risk management or fundamental analysis.

## Live Dashboard Link

**Live App:** *(replace with your deployed Streamlit link)*

[> [https://your-streamlit-app-url.streamlit.app](https://your-streamlit-app-url.streamlit.app)](https://tarmeez-dashboard-3ao3phlcgjmpdore97hdip.streamlit.app/)


## Assumptions & Limitations

This project assumes that historical price patterns and technical indicators contain signal for short-term direction; in reality, financial markets are noisy and influenced by news, macro events, and liquidity. Yahoo Finance data is suitable for prototyping and educational use, but it may not match the latency, coverage, or contractual guarantees of licensed market data providers. The CART model is intentionally interpretable but may underperform more advanced methods on complex patterns. Results should therefore be interpreted as **decision support**, not investment advice.
**References:**

* Market data licensing considerations: [https://www.saudiexchange.sa](https://www.saudiexchange.sa)
* Limits of technical analysis: [https://www.investopedia.com/terms/t/technicalanalysis.asp](https://www.investopedia.com/terms/t/technicalanalysis.asp)

---

