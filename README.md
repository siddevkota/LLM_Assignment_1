# 🧠 KU AICL434 (LLM Assignment 1)

---

## 📁 Project Structure

```plaintext
LLM_Assignment_1/
├── .venv/                     # Python virtual environment (ignored in Git)
├── assignment_1.1/            # Basic NLP tasks
│   ├── __pycache__/
│   ├── api.py                 # Flask/FastAPI backend
│   ├── nlp_preprocessing.py   # Text cleaning and preprocessing
│   └── streamlit_app.py       # Streamlit front-end app
├── assignment_1.2/            # Embedding-based similarity
│   ├── __pycache__/
│   ├── api_embeddings.py      # API to serve vector embeddings
│   ├── embedding_preprocessing.py
│   └── streamlit_embeddings_app.py
├── assignment_1.3/            # Abstractive summarization with Seq2Seq
│   ├── news_summary.csv       # Raw dataset (short)
│   ├── news_summary_more.csv  # Expanded dataset
│   ├── seq2seq_model.h5       # Trained LSTM model
│   ├── summarizer_app.py      # Streamlit app for summary generation
│   ├── train_model.ipynb      # Jupyter notebook for training model
│   ├── training_loss.png      # Training loss curve
│   ├── x_tokenizer.pkl        # Tokenizer for input sequences
│   └── y_tokenizer.pkl        # Tokenizer for summaries
```

---

## ⚙️ Setup Instructions

### ✅ Prerequisites

- **Python 3.8+**  
- **Git**  
- *(Optional but recommended)*: **Virtual environment manager** (e.g., `venv`, `conda`)

### 📥 Installation Steps

```bash
# Clone the repository
git clone <repository-url>
cd LLM_Assignment_1

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # For Unix/macOS
# OR
.venv\Scripts\activate          # For Windows

# Install all dependencies
pip install -r requirements.txt
```

⚠️ **Note:** If you encounter missing packages, re-run:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🧪 For Assignment 1.1: NLP Preprocessing Basics
**Objective:** Apply core NLP preprocessing techniques and demonstrate them via an
interactive web interface.

**Tasks:**
- Implement tokenization, lemmatization, stemming, POS tagging, and Named Entity
-Recognition (NER) using NLTK or spaCy.
- Compare lemmatization and stemming with at least 10 examples and explain the
differences.
- Create a REST API that exposes these preprocessing functions.
- Build a simple demo web app where users can input text and view the processed
output interactively.

```bash
# Terminal 1: Run the backend API
cd assignment_1.1
python api.py
```

```bash
# Terminal 2: Launch the Streamlit app
streamlit run streamlit_app.py
```
![Assignment1.1](screenshots/assignment_1.1.png)

### 🧬 For Assignment 1.2: Word Embeddings & Visualization
**Objective:** Explore and apply word embedding techniques with an interactive component.

**Tasks:**
- Use TF-IDF or GloVe embeddings on a small custom corpus.
- Visualize embeddings using dimensionality reduction techniques like t-SNE or PCA.
- Develop a REST API to compute and return embeddings for input words.
- Create a simple web app that lets users enter words and see their embeddings and
nearest neighbors.

```bash
# Terminal 1: Start the embeddings API
cd assignment_1.2
python api_embeddings.py
```

```bash
# Terminal 2: Open the embeddings UI
streamlit run streamlit_embeddings_app.py
```

![Assignment1.2](screenshots/assignment_1.2.png)

### 📰 For Assignment 1.3: Seq2Seq Summarization with LSTM

**Objective:** Implement and evaluate a basic sequence-to-sequence model for text
summarization.

**Tasks:**
- Build an encoder-decoder model using LSTM layers for abstractive text
summarization.
- Use a small dataset (e.g., 100–200 news articles or custom data).
- Train and evaluate the model.

```bash
# Run directly via Streamlit
cd assignment_1.3
streamlit run summarizer_app.py
```
![Assignment1.3](screenshots/assignment_1.3.png)

---

## 📎 Additional Notes

- Make sure to activate the virtual environment before running any script.  
- To retrain the summarization model, use the `train_model.ipynb` notebook inside `assignment_1.3/`.  
- You can update `requirements.txt` using:

```bash
pip freeze > requirements.txt
```

---



