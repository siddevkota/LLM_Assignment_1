import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Assignment 1.1", layout="centered")
st.title("Assignment 1.1: NLP Preprocessing Basics")

API_URL = "http://127.0.0.1:5000"

menu = st.sidebar.selectbox(
    "Choose an action:", ["Preprocess Text", "Compare Lemmas & Stems"]
)

if menu == "Preprocess Text":
    st.header("1. Preprocessing Text")
    user_text = st.text_area(
        "Enter text to preprocess:",
        value="Apple is looking at buying U.K. startup for $1 billion",
    )

    if st.button("Run Preprocessing"):
        with st.spinner("Fetching data from API..."):
            response = requests.post(f"{API_URL}/preprocess", json={"text": user_text})

        if response.status_code == 200:
            result = response.json()
            st.success("✅ API call successful!")

            st.subheader("Tokens")
            st.write(result["tokens"])

            st.subheader("Lemmas")
            st.write(result["lemmas"])

            st.subheader("Stems")
            st.write(result["stems"])

            st.subheader("Part-of-Speech Tags")
            pos_df = pd.DataFrame(result["pos_tags"], columns=["Token", "POS"])
            st.dataframe(pos_df, use_container_width=True)

            st.subheader("Named Entities")
            if result["entities"]:
                ent_df = pd.DataFrame(result["entities"], columns=["Entity", "Label"])
                st.dataframe(ent_df, use_container_width=True)
            else:
                st.write("No named entities detected.")
        else:
            st.error("Failed to fetch data from API.")

elif menu == "Compare Lemmas & Stems":
    st.header("2. Comparing Lemmatization vs. Stemming")
    default = [
        "running",
        "flies",
        "went",
        "fairly",
        "studies",
        "argued",
        "bigger",
        "cats",
        "played",
        "cities",
    ]
    input_words = st.text_area(
        "Enter words (comma-separated):", value=",".join(default)
    )

    if st.button("Compare"):
        words = [w.strip() for w in input_words.split(",") if w.strip()]
        with st.spinner("Fetching comparison from API..."):
            response = requests.post(f"{API_URL}/compare", json={"words": words})

        if response.status_code == 200:
            table = response.json()
            st.success("✅ API call successful!")
            df = pd.DataFrame(table)
            st.table(df)
            st.markdown(
                "> **DIFFERENCE:** "
                "**Lemmatization** returns linguistically valid base forms, "
                "**Stemming** is a crude rule-based truncation."
            )
        else:
            st.error("Failed to fetch data from API.")
