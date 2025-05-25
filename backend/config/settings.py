import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "backend", "models", "sentiment_model.pkl")

# API settings
API_HOST = "localhost"
API_PORT = 8000

# Frontend settings
FRONTEND_HOST = "localhost"
FRONTEND_PORT = 8501 