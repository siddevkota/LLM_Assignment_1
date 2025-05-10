import streamlit as st


st.set_page_config(page_title="Assignment 1.3", layout="wide")


import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle


DATA_PATH = "news_summary_more.csv"
MODEL_PATH = "seq2seq_model.h5"
X_TOKENIZER_PATH = "x_tokenizer.pkl"
Y_TOKENIZER_PATH = "y_tokenizer.pkl"
MAX_TEXT_LEN = 100
MAX_SUMMARY_LEN = 15


@st.cache_data
def load_dataset():
    df_raw = pd.read_csv(
        DATA_PATH, encoding="iso-8859-1", engine="python", on_bad_lines="skip"
    )
    df = df_raw.iloc[:, :2].copy()
    df.columns = ["summary", "text"]
    return df


df = load_dataset()


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH, compile=False)
    with open(X_TOKENIZER_PATH, "rb") as f_x, open(Y_TOKENIZER_PATH, "rb") as f_y:
        x_tokenizer = pickle.load(f_x)
        y_tokenizer = pickle.load(f_y)
    return model, x_tokenizer, y_tokenizer


model, x_tokenizer, y_tokenizer = load_artifacts()


target_word_index = y_tokenizer.word_index
reverse_target_word_index = {i: w for w, i in target_word_index.items()}
start_token = target_word_index.get("sostok")
end_token = target_word_index.get("eostok")


def preprocess_text(text: str) -> np.ndarray:
    seq = x_tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_TEXT_LEN, padding="post")


def decode_sequence(input_seq: np.ndarray) -> str:
    summary_seq = np.zeros((1, MAX_SUMMARY_LEN), dtype="int32")
    summary_seq[0, 0] = start_token
    for i in range(1, MAX_SUMMARY_LEN):
        preds = model.predict([input_seq, summary_seq], verbose=0)
        sampled_idx = np.argmax(preds[0, i - 1, :])
        summary_seq[0, i] = sampled_idx
        if sampled_idx == end_token:
            break
    words = [
        reverse_target_word_index[idx]
        for idx in summary_seq[0]
        if idx not in (0, start_token, end_token)
    ]
    return " ".join(words)


st.title("Assignment 1.3: Seq2Seq Summarization with LSTM")
st.markdown("Select an article and compare the model's summary against the original.")

# Sidebar selector
choice = st.sidebar.selectbox("Choose an article (ground truth summary)", df["summary"])
idx = df[df["summary"] == choice].index[0]
article = df.at[idx, "text"]

st.subheader("📄 Article Text")
st.write(article)

if st.button("Generate Summary"):
    if not article.strip():
        st.warning("No text available for this article.")
    else:
        with st.spinner("Summarizing..."):
            seq = preprocess_text(article)
            gen = decode_sequence(seq)
        st.subheader("🔖 Generated Summary")
        st.write(gen)
        st.subheader("📝 Original Summary")
        st.write(choice)
