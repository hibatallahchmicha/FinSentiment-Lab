# FinSentiment Lab

A **production-ready financial sentiment analysis platform** combining real-time news sentiment scoring with stock price movements and advanced ML analysis.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit)](https://finsentiment-lab-01.streamlit.app/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-00a651?logo=fastapi)](https://finsentiment-lab.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.12+-3776ab?logo=python)](https://www.python.org/)

---

## Overview

FinSentiment Lab analyzes financial news sentiment and correlates it with stock price movements for **AAPL**, **MSFT**, and **TSLA**. The platform combines:

- **Sentiment Analysis**: Dual-model scoring (FinBERT + Claude AI)
- **Price Analysis**: Real-time stock data from Yahoo Finance
- **Statistical Analysis**: Granger causality, correlation matrices, feature importance
- **Interactive Dashboard**: 6 analytical views with live API integration
- **Production Deployment**: Streamlit Cloud + Render backend

---

## Features

### 📊 Dashboard Views

1. **Sentiment Timeline** - Daily sentiment scores with trend analysis
2. **Price Overlay** - Sentiment vs. stock price movements
3. **Correlation Heatmap** - Cross-ticker sentiment/price correlations
4. **Feature Importance** - ML model feature rankings by ticker
5. **Granger Causality** - Statistical tests for sentiment→price causality
6. **Leaderboard** - Model performance metrics (AUC/F1 scores)

### 🔧 Technical Features

- ✅ Real-time API endpoints for all analysis
- ✅ 5-minute caching for performance
- ✅ Graceful fallback to mock data on API errors
- ✅ Dark theme optimized for financial dashboards
- ✅ Responsive design (desktop/tablet/mobile)
- ✅ CORS-enabled for cross-origin requests

---

## Architecture

```
Frontend (Streamlit Cloud)
        ↓
   Streamlit App
   (6 views + Plotly)
        ↓
FastAPI Backend (Render)
        ↓
Analysis Engines
├─ Sentiment: FinBERT + Claude
├─ Statistics: Granger, Correlation
└─ ML Models: XGBoost, LightGBM, Random Forest
        ↓
Data Sources
├─ News: NewsAPI
├─ Prices: Yahoo Finance
└─ Processed: JSON files
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.28+, Plotly 5.18+ |
| **Backend** | FastAPI 0.111+, Uvicorn |
| **Sentiment** | FinBERT, Anthropic Claude |
| **ML Models** | XGBoost, LightGBM, scikit-learn |
| **Data** | Pandas, NumPy, GPyTorch |
| **Deployment** | Streamlit Cloud, Render |

---

## Quick Start

### Prerequisites
- Python 3.12+
- pip or conda
- NewsAPI key (get free key at [newsapi.org](https://newsapi.org))

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/hibatallahchmicha/FinSentiment-Lab.git
cd FinSentiment-Lab
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure secrets**
```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
api_base = "http://localhost:8000"
newsapi_key = "your_newsapi_key_here"
EOF
```

5. **Start backend API**
```bash
python -m uvicorn main:app --reload --port 8000
```

6. **Start Streamlit dashboard** (new terminal)
```bash
streamlit run streamlit_app.py
```

Dashboard loads at: http://localhost:8503

---

## Deployment

### Option 1: Streamlit Cloud (Frontend)

1. Push to GitHub
2. Go to https://share.streamlit.io
3. "Create app" → Select your GitHub repo
4. Add secrets in Settings:
```toml
api_base = "https://your-backend-url.com"
newsapi_key = "your_newsapi_key"
```

### Option 2: Render (Backend)

1. Create [Render account](https://render.com)
2. Connect GitHub repo
3. Create Web Service with:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Copy deployed URL
5. Update Streamlit secrets with backend URL

---

## Project Structure

```
llm-financial-news-analyzer/
├── streamlit_app.py              # Main dashboard (6 views)
├── main.py                        # FastAPI app entry point
├── requirements.txt               # Python dependencies
│
├── analysis/
│   ├── api_router.py             # FastAPI endpoints
│   ├── pipeline.py               # Analysis orchestration
│   ├── granger.py                # Granger causality tests
│   ├── correlation.py            # Correlation matrices
│   ├── regression.py             # Regression models
│   └── __pycache__/
│
├── data_collection/
│   ├── pipeline.py               # Data fetching orchestration
│   ├── news/
│   │   └── newsapi_fetcher.py    # NewsAPI integration
│   └── prices/
│       └── yfinance_fetcher.py   # Yahoo Finance integration
│
├── feature_engineering/
│   ├── pipeline.py               # Feature creation
│   ├── sentiment_features.py     # Sentiment-derived features
│   ├── momentum_features.py      # Technical indicators
│   └── volatility_features.py    # Volatility metrics
│
├── sentiment_engine/
│   ├── pipeline.py               # Sentiment orchestration
│   ├── finbert_scorer.py         # FinBERT model
│   ├── claude_scorer.py          # Claude API integration
│   └── aggregator.py             # Score aggregation
│
├── models/
│   ├── pipeline.py               # Model training/evaluation
│   ├── predictors.py             # Model definitions
│   ├── preparation.py            # Data preprocessing
│   └── evaluation.py             # Model metrics
│
├── data/
│   ├── raw_news/                 # Raw news articles
│   ├── raw_prices/               # Raw price data
│   ├── cache/                    # Cached API responses
│   └── processed/                # Final analysis outputs
│
├── .streamlit/
│   ├── config.toml               # UI theme & settings
│   └── secrets.toml              # API keys (git-ignored)
│
├── DEPLOYMENT.md                 # Deployment guide
├── README.md                     # This file
└── .gitignore                    # Exclude secrets & cache
```

---

## API Endpoints

All endpoints available at `https://finsentiment-lab.onrender.com/docs`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analysis/sentiment/{ticker}` | GET | Daily sentiment scores |
| `/analysis/leaderboard` | GET | Model performance metrics |
| `/analysis/features` | GET | Feature importance by model |
| `/analysis/granger` | GET | Granger causality results |
| `/analysis/correlation` | GET | Correlation matrix |
| `/analysis/health` | GET | Service health check |

### Example Request
```bash
curl https://finsentiment-lab.onrender.com/analysis/sentiment/AAPL?days=60
```

---

## Configuration

### Environment Variables

Create `.streamlit/secrets.toml` for local development:

```toml
# API Configuration
api_base = "http://localhost:8000"  # Backend URL

# Data Sources
newsapi_key = "your_key_here"       # From newsapi.org
```

### Streamlit Settings

Dashboard theme in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#00d4ff"           # Cyan
backgroundColor = "#080c14"        # Dark navy
secondaryBackgroundColor = "#0d1526"
textColor = "#e8f0fe"              # Light blue

[client]
showErrorDetails = false
```

---

## Performance

- **Frontend Response**: < 2s (cached)
- **API Response**: < 500ms (JSON from files)
- **Cache Duration**: 5 minutes
- **Concurrent Users**: Unlimited on Streamlit Cloud + Render

---

## Troubleshooting

### API Error: "Connection refused"
- Ensure backend is running: `https://finsentiment-lab.onrender.com/docs`
- Check Streamlit secrets have correct `api_base` URL
- Restart Streamlit with: Ctrl+R

### No data displayed
- Check backend API is responding: `curl https://finsentiment-lab.onrender.com/analysis/health`
- Verify data files exist in `data/processed/`
- Clear Streamlit cache: Menu → Clear cache

### Build timeout on Streamlit Cloud
- Ensure `requirements.txt` is minimal (no torch/transformers)
- Check `.streamlit/config.toml` has no [server] section
- Remove unused dependencies

---

## Data Sources

| Source | Usage | Frequency |
|--------|-------|-----------|
| **NewsAPI** | Financial news articles | Real-time |
| **Yahoo Finance** | Stock prices (OHLCV) | Daily |
| **Internal ML** | Sentiment scores | Daily |
| **Processed JSON** | Historical analysis | Cached |

---

## Models & Algorithms

### Sentiment Scoring
- **FinBERT**: Fine-tuned BERT for financial sentiment
- **Claude AI**: Advanced semantic understanding
- **Aggregation**: Weighted ensemble (60% FinBERT + 40% Claude)

### Predictive Models
- **XGBoost**: Primary price predictor
- **LightGBM**: Fast alternative model
- **Random Forest**: Ensemble baseline

### Statistical Tests
- **Granger Causality**: Sentiment → Price causality
- **Pearson Correlation**: Cross-ticker relationships
- **Feature Importance**: SHAP values from models

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open Pull Request

---

## License

MIT License - see LICENSE file for details

---

## Author

**Hibatallah Chmicha**
- GitHub: [@hibatallahchmicha](https://github.com/hibatallahchmicha)


---

## Acknowledgments

- **NewsAPI** for financial news data
- **Yahoo Finance** for stock prices
- **Hugging Face** for FinBERT model
- **Anthropic** for Claude AI
- **Streamlit** & **FastAPI** for excellent frameworks

---

## Live Demo

- **Dashboard**: https://finsentiment-lab-01.streamlit.app/
- **API Docs**: https://finsentiment-lab.onrender.com/docs
- **GitHub**: https://github.com/hibatallahchmicha/FinSentiment-Lab

---

**Last Updated**: March 11, 2026
