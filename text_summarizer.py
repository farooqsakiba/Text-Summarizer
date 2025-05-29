# text_summarizer.py

import streamlit as st
from transformers import pipeline

# Load the summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# Streamlit UI
st.title("📝 Text Summarizer using DistilBART (via HuggingFace)")
st.markdown("Built by **Sakiba Farooq**")

text_input = st.text_area("Enter the text you want to summarize:", height=300)

if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer(text_input, max_length=130, min_length=30, do_sample=False)
            st.subheader("🔽 Summary:")
            st.success(summary[0]['summary_text'])
