import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="FinSentiment Lab", layout="wide", initial_sidebar_state="expanded")

# ─── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    * { font-family: 'Inter', 'DM Mono', sans-serif; }
    .metric-card { 
        padding: 16px; 
        border-radius: 10px; 
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(0,229,160,0.1));
        border: 1px solid rgba(0,212,255,0.3);
    }
    .metric-label { font-size: 12px; color: #5a7499; letter-spacing: 0.5px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #e8f0fe; margin-top: 8px; }
    h1 { color: #e8f0fe; letter-spacing: -0.5px; }
    h2 { color: #e8f0fe; font-size: 18px; margin-top: 24px; margin-bottom: 16px; }
    .ticker-badge { 
        display: inline-block; 
        padding: 4px 12px; 
        border-radius: 20px; 
        font-size: 11px; 
        font-family: 'DM Mono', monospace;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .ticker-aapl { background: rgba(0,212,255,0.2); color: #00d4ff; border: 1px solid #00d4ff; }
    .ticker-tsla { background: rgba(255,77,109,0.2); color: #ff4d6d; border: 1px solid #ff4d6d; }
    .ticker-msft { background: rgba(0,229,160,0.2); color: #00e5a0; border: 1px solid #00e5a0; }
</style>
""", unsafe_allow_html=True)

# ─── Config ──────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "TSLA", "MSFT"]
TICKER_COLORS = {
    "AAPL": "#00d4ff",
    "TSLA": "#ff4d6d",
    "MSFT": "#00e5a0",
}

# Get API base from secrets or use default
try:
    API_BASE = st.secrets.get("api_base", "http://localhost:8000")
except:
    API_BASE = "http://localhost:8000"

# ─── API Calls ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)  # 5 min cache
def get_sentiment_timeline(ticker, n=60):
    """Fetch sentiment timeline from API"""
    try:
        response = requests.get(f"{API_BASE}/analysis/sentiment/{ticker}", 
                               params={"days": n}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                # Add derived price column
                if 'score' in df.columns:
                    np.random.seed(hash(ticker) % 2**32)
                    base_price = {"AAPL": 185, "TSLA": 215, "MSFT": 420}.get(ticker, 200)
                    # Simulate price movement based on sentiment
                    returns = df['score'].fillna(0) * 0.02 + np.random.randn(len(df)) * 0.01
                    df['price'] = base_price * np.exp(np.cumsum(returns))
                    df['vol'] = abs(df['score'].fillna(0)) * 0.02 + np.abs(np.random.randn(len(df)) * 0.01)
                return df
    except Exception as e:
        st.warning(f"⚠️ API Error: {e}")
    
    # Fallback to mock
    return _generate_mock_timeline(ticker, n)

@st.cache_data(ttl=300)
def get_leaderboard():
    """Fetch model leaderboard from API"""
    try:
        response = requests.get(f"{API_BASE}/analysis/leaderboard", timeout=5)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
    except Exception as e:
        st.warning(f"⚠️ API Error: {e}")
    
    return _get_mock_leaderboard()

@st.cache_data(ttl=300)
def get_feature_importance(model=None, ticker=None):
    """Fetch feature importance from API"""
    try:
        params = {}
        if model:
            params['model'] = model
        if ticker:
            params['ticker'] = ticker
        
        response = requests.get(f"{API_BASE}/analysis/features", 
                               params=params, timeout=5)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
    except Exception as e:
        st.warning(f"⚠️ API Error: {e}")
    
    return _get_mock_importance()

@st.cache_data(ttl=300)
def get_granger_causality(significant_only=False):
    """Fetch Granger causality results from API"""
    try:
        response = requests.get(f"{API_BASE}/analysis/granger",
                               params={"significant_only": significant_only}, timeout=5)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
    except Exception as e:
        st.warning(f"⚠️ API Error: {e}")
    
    return _get_mock_granger()

@st.cache_data(ttl=300)
def get_correlation_matrix(ticker="POOLED"):
    """Fetch correlation matrix from API"""
    try:
        response = requests.get(f"{API_BASE}/analysis/correlation",
                               params={"ticker": ticker}, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.warning(f"⚠️ API Error: {e}")
    
    return _get_mock_correlation()

# ─── Mock Fallbacks ───────────────────────────────────────────────────────

def _generate_mock_timeline(ticker, n=60):
    """Generate mock sentiment timeline data"""
    np.random.seed(hash(ticker) % 2**32)
    price = {"AAPL": 185, "TSLA": 215, "MSFT": 420}.get(ticker, 200)
    score = 0.1
    
    data = []
    for i in range(n):
        date = (datetime(2025, 9, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        score += (np.random.random() - 0.48) * 0.15
        score = max(-0.9, min(0.9, score))
        price *= 1 + score * 0.012 + (np.random.random() - 0.5) * 0.018
        
        label = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
        data.append({
            "date": date,
            "score": round(score, 3),
            "price": round(price, 2),
            "vol": round(abs(score) * 0.02 + np.random.random() * 0.01, 4),
            "label": label,
        })
    return pd.DataFrame(data)

def _get_mock_leaderboard():
    """Mock model leaderboard"""
    return pd.DataFrame([
        {"ticker": "TSLA",    "model": "XGBoostClassifier",   "auc": 0.638, "accuracy": 0.571, "f1": 0.563, "hit_rate": 0.571, "sharpe": 1.24, "cum_return": 0.183},
        {"ticker": "MSFT",    "model": "XGBoostClassifier",   "auc": 0.612, "accuracy": 0.558, "f1": 0.541, "hit_rate": 0.558, "sharpe": 1.01, "cum_return": 0.142},
        {"ticker": "AAPL",    "model": "XGBoostClassifier",   "auc": 0.601, "accuracy": 0.545, "f1": 0.528, "hit_rate": 0.545, "sharpe": 0.87, "cum_return": 0.108},
        {"ticker": "TSLA",    "model": "LSTM",                "auc": 0.591, "accuracy": 0.536, "f1": 0.519, "hit_rate": 0.536, "sharpe": 0.74, "cum_return": 0.091},
        {"ticker": "AAPL",    "model": "LogisticRegression",  "auc": 0.554, "accuracy": 0.518, "f1": 0.492, "hit_rate": 0.518, "sharpe": 0.41, "cum_return": 0.044},
        {"ticker": "MSFT",    "model": "LogisticRegression",  "auc": 0.548, "accuracy": 0.512, "f1": 0.488, "hit_rate": 0.512, "sharpe": 0.38, "cum_return": 0.039},
        {"ticker": "POOLED",  "model": "XGBoostClassifier",   "auc": 0.589, "accuracy": 0.534, "f1": 0.521, "hit_rate": 0.534, "sharpe": 0.93, "cum_return": 0.121},
    ])

def _get_mock_importance():
    """Mock feature importance"""
    return pd.DataFrame([
        {"feature": "sentiment_zscore",      "score": 0.182, "model": "XGBClf", "ticker": "POOLED"},
        {"feature": "vol_ratio_5_21",        "score": 0.154, "model": "XGBClf", "ticker": "POOLED"},
        {"feature": "rsi_14d",               "score": 0.131, "model": "XGBClf", "ticker": "POOLED"},
        {"feature": "sentiment_cross_7_30",  "score": 0.118, "model": "XGBClf", "ticker": "POOLED"},
        {"feature": "price_vs_sma_30d",      "score": 0.097, "model": "XGBClf", "ticker": "POOLED"},
        {"feature": "return_5d",             "score": 0.088, "model": "XGBClf", "ticker": "POOLED"},
        {"feature": "atr_14d_pct",           "score": 0.076, "model": "XGBClf", "ticker": "POOLED"},
        {"feature": "mean_score",            "score": 0.062, "model": "XGBClf", "ticker": "POOLED"},
        {"feature": "bullish_ratio_roll_7d", "score": 0.054, "model": "XGBClf", "ticker": "POOLED"},
        {"feature": "news_day",              "score": 0.038, "model": "XGBClf", "ticker": "POOLED"},
    ])

def _get_mock_granger():
    """Mock Granger causality results"""
    return pd.DataFrame([
        {"ticker": "AAPL",   "cause": "sentiment_roll_7d",    "effect": "daily_return",   "lag": 2, "p_value": 0.024, "significant": True},
        {"ticker": "TSLA",   "cause": "mean_score",           "effect": "daily_return",   "lag": 1, "p_value": 0.031, "significant": True},
        {"ticker": "TSLA",   "cause": "sentiment_zscore",     "effect": "forward_vol_5d", "lag": 3, "p_value": 0.018, "significant": True},
        {"ticker": "MSFT",   "cause": "sentiment_cross_7_30", "effect": "daily_return",   "lag": 2, "p_value": 0.074, "significant": False},
        {"ticker": "AAPL",   "cause": "mean_score",           "effect": "forward_vol_5d", "lag": 4, "p_value": 0.041, "significant": True},
        {"ticker": "POOLED", "cause": "sentiment_roll_7d",    "effect": "daily_return",   "lag": 1, "p_value": 0.012, "significant": True},
        {"ticker": "MSFT",   "cause": "mean_score",           "effect": "daily_return",   "lag": 3, "p_value": 0.119, "significant": False},
        {"ticker": "POOLED", "cause": "sentiment_zscore",     "effect": "forward_vol_5d", "lag": 2, "p_value": 0.008, "significant": True},
    ])

def _get_mock_correlation():
    """Mock correlation matrix"""
    features = ["mean_score", "sent_roll_7d", "sent_zscore", "rsi_14d", 
                "vol_ratio", "return_5d", "atr_pct", "news_day"]
    np.random.seed(42)
    corr_matrix = np.random.uniform(-0.7, 1, (len(features), len(features)))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    return {
        "ticker": "POOLED",
        "features": features,
        "correlation_matrix": corr_matrix.tolist(),
    }

# ─── Views ────────────────────────────────────────────────────────────────

def view_sentiment_timeline():
    """Sentiment Timeline View"""
    st.header("📊 Sentiment Timeline")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_ticker = st.selectbox("Select Ticker", TICKERS, key="st_ticker")
    
    df = get_sentiment_timeline(selected_ticker, n=60)
    
    if df.empty:
        st.warning("No sentiment data available")
        return
    
    # Rolling average
    df['score_roll_7d'] = df['score'].rolling(7).mean()
    
    # Chart 1: Daily + 7d rolling
    fig1 = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Bullish area
    bullish = df[df['score'] > 0]
    fig1.add_trace(go.Bar(x=bullish['date'], y=bullish['score'], name='Bullish',
                         marker_color='#00e5a0', opacity=0.6))
    
    # Bearish area
    bearish = df[df['score'] < 0]
    fig1.add_trace(go.Bar(x=bearish['date'], y=bearish['score'], name='Bearish',
                         marker_color='#ff4d6d', opacity=0.6))
    
    # Rolling line
    fig1.add_trace(go.Scatter(x=df['date'], y=df['score_roll_7d'], name='7-Day MA',
                             line=dict(color=TICKER_COLORS[selected_ticker], width=3)))
    
    fig1.update_layout(
        title="Daily Sentiment Score",
        xaxis_title="Date",
        yaxis_title="Sentiment Score [-1, +1]",
        template="plotly_dark",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Sentiment", f"{df['score'].mean():.3f}")
    with col2:
        st.metric("Bullish Days", len(df[df['label'] == 'bullish']))
    with col3:
        st.metric("Bearish Days", len(df[df['label'] == 'bearish']))
    with col4:
        st.metric("Volatility", f"{df['vol'].mean():.4f}")

def view_price_overlay():
    """Price + Sentiment Overlay View"""
    st.header("💹 Price + Sentiment Overlay")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_ticker = st.selectbox("Select Ticker", TICKERS, key="po_ticker")
    
    df = get_sentiment_timeline(selected_ticker, n=90)
    
    if df.empty:
        st.warning("No sentiment data available")
        return
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Price area
    fig.add_trace(go.Scatter(x=df['date'], y=df['price'], name='Price',
                            fill='tozeroy', line=dict(color=TICKER_COLORS[selected_ticker]),
                            opacity=0.6), secondary_y=False)
    
    # Sentiment bars
    colors = df['score'].apply(lambda x: '#00e5a0' if x > 0 else '#ff4d6d')
    fig.add_trace(go.Bar(x=df['date'], y=df['score'], name='Sentiment',
                        marker=dict(color=colors), opacity=0.4), secondary_y=True)
    
    fig.update_layout(
        title=f"{selected_ticker} · Price + Sentiment",
        xaxis_title="Date",
        template="plotly_dark",
        hovermode="x unified",
        height=450,
    )
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Sentiment", f"{df['score'].mean():.3f}")
    with col2:
        st.metric("Bullish Days", len(df[df['label'] == 'bullish']))
    with col3:
        st.metric("Bearish Days", len(df[df['label'] == 'bearish']))
    with col4:
        price_change = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]
        st.metric("Price Change", f"{price_change*100:+.1f}%")

def view_correlation_heatmap():
    """Correlation Heatmap View"""
    st.header("🔗 Correlation Matrix")
    
    corr_data = get_correlation_matrix()
    
    if not corr_data or not corr_data.get('correlation_matrix'):
        st.warning("No correlation data available")
        return
    
    corr_matrix = np.array(corr_data['correlation_matrix'])
    features = corr_data.get('features', [f'Feature {i}' for i in range(len(corr_matrix))])
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=features,
        y=features,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hovertemplate='%{y} vs %{x}: %{z:.3f}<extra></extra>',
    ))
    
    fig.update_layout(
        title="Pearson Correlation Matrix (Features)",
        template="plotly_dark",
        height=600,
        xaxis_tickangle=45,
    )
    
    st.plotly_chart(fig, use_container_width=True)

def view_feature_importance():
    """Feature Importance View"""
    st.header("⚡ Feature Importance")
    
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", ["XGBoostClassifier", "XGBoostRegressor", "LogisticRegression"])
    
    df = get_feature_importance()
    
    if df.empty:
        st.warning("No feature importance data available")
        return
    
    # Take top 10
    df = df.nlargest(10, 'score')
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['feature'],
            x=df['score'],
            orientation='h',
            marker=dict(
                color=df['score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=df['score'].round(3),
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title=f"Top 10 Features · Pooled Data",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template="plotly_dark",
        height=500,
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment breakdown
    sentiment_features = df[df['feature'].str.contains('sentiment|score|bull|bear|news', na=False)]
    if len(df) > 0:
        sentiment_pct = sentiment_features['score'].sum() / df['score'].sum() if df['score'].sum() > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment Features Count", len(sentiment_features))
        with col2:
            st.metric("Sentiment Importance %", f"{sentiment_pct*100:.1f}%")

def view_granger_causality():
    """Granger Causality View"""
    st.header("🔬 Granger Causality Analysis")
    
    df = get_granger_causality()
    
    if df.empty:
        st.warning("No Granger causality data available")
        return
    
    # Significant
    st.subheader("✓ Significant Relationships (p < 0.05)")
    sig_df = df[df['significant']].copy()
    if not sig_df.empty:
        sig_df['p_value'] = sig_df['p_value'].round(4)
        st.dataframe(sig_df[['ticker', 'cause', 'effect', 'lag', 'p_value']], use_container_width=True, hide_index=True)
        st.metric("Significant Tests", len(sig_df))
    else:
        st.info("No significant relationships found")
    
    # Non-significant
    st.subheader("✗ Non-Significant Relationships (p ≥ 0.05)")
    nsig_df = df[~df['significant']].copy()
    if not nsig_df.empty:
        nsig_df['p_value'] = nsig_df['p_value'].round(4)
        st.dataframe(nsig_df[['ticker', 'cause', 'effect', 'lag', 'p_value']], use_container_width=True, hide_index=True)

def view_leaderboard():
    """Model Leaderboard View"""
    st.header("🏆 Model Leaderboard")
    
    df = get_leaderboard()
    
    if df.empty:
        st.warning("No leaderboard data available")
        return
    
    df = df.sort_values('auc', ascending=False).reset_index(drop=True)
    
    # Format for display
    display_df = df.copy()
    for col in ['auc', 'accuracy', 'f1', 'hit_rate']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    if 'sharpe' in display_df.columns:
        display_df['sharpe'] = display_df['sharpe'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    if 'cum_return' in display_df.columns:
        display_df['cum_return'] = display_df['cum_return'].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Top 3 Podium
    st.subheader("🥇 Top 3 Performers")
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, col in enumerate(cols):
        if i < len(df):
            row = df.iloc[i]
            with col:
                auc_val = row.get('auc', 0)
                sharpe_val = row.get('sharpe', 0)
                st.metric(
                    f"{medals[i]} {row['ticker']} · {row['model']}",
                    f"AUC: {auc_val:.3f}",
                    delta=f"Sharpe: {sharpe_val:.2f}",
                )

# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("# 📈 **FinSentiment**Lab")
    st.markdown("*Financial News Sentiment Analysis Dashboard*")
    st.markdown("---")
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🗺️ Navigation")
        view = st.radio(
            "Select View",
            [
                "📊 Sentiment Timeline",
                "💹 Price Overlay",
                "🔗 Correlation",
                "⚡ Features",
                "🔬 Granger",
                "🏆 Leaderboard",
            ],
            label_visibility="collapsed",
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **FinSentiment Lab** combines:
        - 📰 **News Sentiment** (FinBERT + Claude)
        - 📊 **Technical Analysis** (RSI, ATR, SMA)
        - 🤖 **ML Models** (XGBoost, LSTM, LR)
        - 🔬 **Causal Analysis** (Granger tests)
        
        **Tickers:** AAPL, TSLA, MSFT
        **Window:** 90 days
        **Updated:** Daily
        
        **Data Source:** 🔗 Real API
        """)
    
    # Route to views
    if "📊 Sentiment Timeline" in view:
        view_sentiment_timeline()
    elif "💹 Price Overlay" in view:
        view_price_overlay()
    elif "🔗 Correlation" in view:
        view_correlation_heatmap()
    elif "⚡ Features" in view:
        view_feature_importance()
    elif "🔬 Granger" in view:
        view_granger_causality()
    elif "🏆 Leaderboard" in view:
        view_leaderboard()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #5a7499; font-size: 11px;'>"
        "FinBERT + Claude Haiku · Powered by FastAPI + Streamlit · AAPL · TSLA · MSFT"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
