# Sentiment Analysis (IMDB Movie Reviews)

Classifies movie reviews as Positive or Negative using TF-IDF + Logistic Regression (sklearn).

## Results
- Accuracy: 88.6%
- Trained on 20,000 samples from the IMDB 50K Movie Reviews dataset

## Setup
1. Download "IMDB Dataset of 50K Movie Reviews" from Kaggle
2. Place it in this folder, named `IMDB_Dataset.csv`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python Sentiment_Model.py`

## Approach
- Cleaned HTML tags and punctuation from raw text
- Kept negation words (not, no, never) that sklearn's default stopword list removes — negation is critical for sentiment
- TF-IDF vectorization with unigrams + bigrams (max_features=15000)
- Logistic Regression classifier
