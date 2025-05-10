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

```bash
# Terminal 1: Run the backend API
cd assignment_1.1
python api.py
```

```bash
# Terminal 2: Launch the Streamlit app
streamlit run streamlit_app.py
```

### 🧬 For Assignment 1.2: Word Embeddings & Visualization

```bash
# Terminal 1: Start the embeddings API
cd assignment_1.2
python api_embeddings.py
```

```bash
# Terminal 2: Open the embeddings UI
streamlit run streamlit_embeddings_app.py
```

### 📰 For Assignment 1.3: Seq2Seq Summarization with LSTM

```bash
# Run directly via Streamlit
cd assignment_1.3
streamlit run summarizer_app.py
```


---

## 📎 Additional Notes

- Make sure to activate the virtual environment before running any script.  
- To retrain the summarization model, use the `train_model.ipynb` notebook inside `assignment_1.3/`.  
- You can update `requirements.txt` using:

```bash
pip freeze > requirements.txt
```

---

## 🖼️ Screenshots

| Streamlit App | Description |
|---------------|-------------|
| `screenshots/assignment_1.1.png` | Assignment 1.1 Web App demo |
| `screenshots/assignment_1.2.png`      | Assignment 1.2 Web App demo |
| `screenshots/assignment_1.3.png`      | Assignment 1.3 Web App demo |


---

