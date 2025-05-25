import pandas as pd
import numpy as np
from pathlib import Path
import logging
import requests
import tarfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_and_extract_imdb(self, url: str, filename: str) -> Path:
        """
        Download and extract IMDB dataset from URL
        Args:
            url (str): URL to download from
            filename (str): Name to save the file as
        Returns:
            Path: Path to extracted directory
        """
        try:
            archive_path = self.data_dir / filename
            extract_path = self.data_dir / "aclImdb"
            if extract_path.exists():
                logger.info(f"IMDB dataset already extracted at {extract_path}")
                return extract_path
            if not archive_path.exists():
                logger.info(f"Downloading IMDB dataset from {url}")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(archive_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded IMDB dataset to {archive_path}")
            # Extract
            logger.info(f"Extracting {archive_path}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=self.data_dir)
            logger.info(f"Extracted IMDB dataset to {extract_path}")
            return extract_path
        except Exception as e:
            logger.error(f"Error downloading or extracting IMDB dataset: {str(e)}")
            raise

    def process_imdb_reviews(self, imdb_dir: Path, n: int = 10000) -> pd.DataFrame:
        """
        Process IMDB reviews from extracted directory
        Args:
            imdb_dir (Path): Path to extracted IMDB directory
            n (int): Number of reviews to process (total)
        Returns:
            pd.DataFrame: Processed dataframe
        """
        try:
            logger.info("Processing IMDB reviews dataset")
            reviews = []
            sentiments = []
            # Read positive reviews
            pos_dir = imdb_dir / "train" / "pos"
            neg_dir = imdb_dir / "train" / "neg"
            for label, directory in [(1, pos_dir), (0, neg_dir)]:
                files = list(directory.glob("*.txt"))[:n//2]
                for file in files:
                    text = file.read_text(encoding="utf-8", errors="ignore").strip().lower()
                    reviews.append(text)
                    sentiments.append(label)
            df = pd.DataFrame({"review": reviews, "sentiment": sentiments})
            # Shuffle
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            # Save processed data
            processed_path = self.data_dir / 'processed_reviews.csv'
            df.to_csv(processed_path, index=False)
            logger.info(f"Processed data saved to {processed_path}")
            return df
        except Exception as e:
            logger.error(f"Error processing IMDB dataset: {str(e)}")
            raise

    def get_training_data(self) -> tuple:
        """
        Get processed training data
        Returns:
            tuple: (texts, labels)
        """
        processed_path = self.data_dir / 'processed_reviews.csv'
        if not processed_path.exists():
            url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            imdb_dir = self.download_and_extract_imdb(url, "aclImdb_v1.tar.gz")
            df = self.process_imdb_reviews(imdb_dir)
        else:
            df = pd.read_csv(processed_path)
        return df['review'].tolist(), df['sentiment'].tolist()

def main():
    processor = DataProcessor()
    texts, labels = processor.get_training_data()
    logger.info(f"Loaded {len(texts)} reviews for training")
    
if __name__ == "__main__":
    main() 