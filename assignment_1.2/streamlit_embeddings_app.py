import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

st.set_page_config(page_title="Assignment 1.2", layout="centered")
st.title("Assignment 1.2: Word Embeddings & Visualization")

API_URL = "http://127.0.0.1:5000"


word = st.text_input("Enter a word to embed:", "king")
topn = st.slider("Number of nearest neighbors:", 1, 20, 5)
method = st.selectbox("Dimensionality reduction method:", ["PCA", "t-SNE"])


perplexity = None
if method == "t-SNE":
    n_samples = topn + 1
    max_perp = max(1, n_samples - 1)
    default_perp = min(30, max_perp)
    perplexity = st.slider(
        f"t-SNE Perplexity (must be < {n_samples}):", 1, max_perp, default_perp
    )

if st.button("Compute and Visualize"):

    with st.spinner("Fetching embedding..."):
        resp1 = requests.post(f"{API_URL}/embed", json={"word": word})
    if resp1.status_code != 200:
        st.error(f"Error: {resp1.json().get('error')}")
        st.stop()
    emb_main = resp1.json()["embedding"]
    st.success("✅ Embedding fetched for main word")

    with st.spinner("Fetching neighbors..."):
        resp2 = requests.post(f"{API_URL}/neighbors", json={"word": word, "topn": topn})
    if resp2.status_code != 200:
        st.error(f"Error: {resp2.json().get('error')}")
        st.stop()
    neighbors = resp2.json()["neighbors"]
    st.success("✅ Neighbors fetched")

    words = [word] + [n["word"] for n in neighbors]
    vectors = [emb_main]
    for n in neighbors:
        r = requests.post(f"{API_URL}/embed", json={"word": n["word"]})
        vectors.append(r.json()["embedding"])

    arr = np.array(vectors)

    if method == "PCA":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(
            n_components=2,
            init="random",
            random_state=0,
            perplexity=perplexity,
        )

    try:
        coords = reducer.fit_transform(arr)
    except ValueError as e:
        st.error(f"Error during dimensionality reduction: {e}")
        st.stop()

    fig, ax = plt.subplots()
    for i, w in enumerate(words):
        ax.scatter(coords[i, 0], coords[i, 1])
        ax.annotate(w, (coords[i, 0], coords[i, 1]))
    st.pyplot(fig)

    st.subheader("Nearest Neighbors")
    neigh_df = pd.DataFrame(neighbors)
    st.dataframe(neigh_df, use_container_width=True)

    st.subheader("Main Word Embedding (first 10 dimensions)")
    st.write(emb_main[:10], "...")
