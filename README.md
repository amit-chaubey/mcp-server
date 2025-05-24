# MCP Sentiment Inference API

A Machine Learning Inference API using MCP server architecture for sentiment analysis.

## Project Structure

```
mcp-sentiment-api/
│
├── backend/
│   ├── main.py
│   ├── services/
│   │   ├── inference.py
│   │   └── model_loader.py
│   ├── models/
│   │   └── sentiment_model.pkl
│   └── config/
│       └── settings.py
│
├── frontend/
│   └── app.py
│
├── data/
│   └── sample_text.csv
│
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI backend:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the Streamlit frontend:
```bash
cd frontend
streamlit run app.py
```

## API Endpoints

- POST `/predict`: Predict sentiment for given text
  - Request body: `{"text": "your text here"}`
  - Response: `{"sentiment": "positive" | "negative"}`

## Technologies Used

- FastAPI for backend API
- Streamlit for frontend UI
- Scikit-learn for ML model
- Joblib for model persistence 