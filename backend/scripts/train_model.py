import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config.settings import MODEL_PATH
from backend.scripts.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample training data"""
    texts = [
        "I love this product, it's amazing!",
        "This is the best thing I've ever bought",
        "Absolutely fantastic experience",
        "Great service and fast delivery",
        "Highly recommended",
        "This is terrible, I want a refund",
        "Worst experience ever",
        "Complete waste of money",
        "I'm very disappointed",
        "Would not recommend to anyone"
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 for positive, 0 for negative
    return texts, labels

def train_model():
    """Train and save the sentiment analysis model"""
    try:
        # Get processed data
        processor = DataProcessor()
        texts, labels = processor.get_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training data size: {len(X_train)}")
        logger.info(f"Testing data size: {len(X_test)}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ])
        
        # Define parameter grid for tuning
        param_grid = {
            'vectorizer__max_features': [3000, 5000, 7000],
            'classifier__C': [0.1, 1.0, 10.0]
        }
        
        # Grid search for best parameters
        logger.info("Starting grid search for best parameters...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.3f}")
        logger.info("\nClassification Report:")
        logger.info(report)
        
        # Create models directory if it doesn't exist
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(best_model, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        
        # Save model metadata
        metadata = {
            'accuracy': accuracy,
            'best_params': grid_search.best_params_,
            'feature_names': best_model.named_steps['vectorizer'].get_feature_names_out().tolist()
        }
        joblib.dump(metadata, str(Path(MODEL_PATH).with_suffix('.metadata.pkl')))
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 