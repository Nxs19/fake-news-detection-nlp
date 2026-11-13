# Comparative Text Classification: Fake News Detection

A comparative Natural Language Processing (NLP) study contrasting a traditional **Statistical Model (TF-IDF + Logistic Regression)** against a **Deep Learning Architecture (1D-CNN + pre-trained GloVe embeddings)** on the ISOT Fake News Dataset (~44,900 full-text news articles).

Developed for **CM3060 Natural Language Processing** by **Ng Xin Sheng**.

---

## Project Overview & Research Question

While deep learning models capture local phrase context (n-grams) and semantic similarity, they incur substantial memory and training overhead compared to linear bag-of-words baselines. This project evaluates whether the accuracy gain of a 1D Convolutional Neural Network justifies its ~50x higher compute cost over Logistic Regression when stylistic cues are strong.

### Key Engineering Highlights
1. **Source Sanitization (Data Leakage Prevention):** Programmatically strips `WASHINGTON (Reuters)` bylines, `\breuters\b` tokens, and URLs prior to feature extraction so models classify based on intrinsic linguistic style rather than publisher metadata.
2. **Dual Preprocessing Pipeline:**
   - *Statistical Path (Aggressive):* Lowercasing, punctuation removal, English stopword filtering, and WordNet lemmatization to minimize TF-IDF feature sparsity (`max_features=5000`).
   - *Deep Learning Path (Gentle):* Lowercasing and URL/source stripping while retaining stopwords and sentence punctuation (`.,!?`) to preserve local n-gram syntax (`vocab_size=20000`, `max_length=200`).
3. **Index-Locked Data Alignment:** Splits on DataFrame indices (`df.index`) before retrieving processed text features, eliminating feature-label misalignment across parallel preprocessing pipelines.

---

## Experimental Results (80/20 Train-Test Split — 8,980 Test Articles)

| Model | Features / Embeddings | Accuracy | Precision | Recall | F1-Score | Relative Training Cost |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | TF-IDF (Top 5,000 unigrams, lemmatized) | **98.07%** | 0.98 | 0.98 | 0.98 | Seconds (CPU) |
| **1D-CNN (Deep Learning)** | Frozen GloVe `6B.100d` + `Conv1D(128, k=5)` | **98.93%** | 0.99 | 0.99 | 0.99 | Minutes (~50x slower, 400k vectors) |

* **Key Takeaway:** The 1D-CNN reduces classification errors from 173/8,980 (Logistic Regression) down to 96/8,980 (+0.86% accuracy gain), demonstrating diminishing returns where Logistic Regression is preferable for low-latency CPU deployment and the 1D-CNN is suited for high-recall content moderation.

---

## Repository Structure

```text
fake-news-detection-nlp/
├── fake_news_detection.ipynb   # Full notebook with code, training logs, and inline plots
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes GloVe embeddings (>100 MB) and raw CSV datasets
└── README.md                   # Project documentation and setup guide
```

*(Local-only files required to re-run the notebook from scratch: `True.csv`, `Fake.csv`, and `glove.6B.100d.txt`.)*

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Nxs19/fake-news-detection-nlp.git](https://github.com/Nxs19/fake-news-detection-nlp.git)
   cd fake-news-detection-nlp
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Python dependencies and NLTK corpora:**
   ```bash
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. **Download external dataset and embeddings into the project root:**
   - **ISOT Fake News Dataset:** Download `True.csv` and `Fake.csv` from the [University of Victoria ISOT Research Lab](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) (or Kaggle) and place both files in the root folder.
   - **GloVe Word Embeddings:** Download `glove.6B.zip` from [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/), extract `glove.6B.100d.txt`, and place it in the root folder.

5. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```