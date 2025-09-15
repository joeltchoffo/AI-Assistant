import streamlit as st
from rag.search import search

st.set_page_config(page_title="RAG Mini", layout="wide")
st.title("🔎 RAG Mini • FAISS (Phase 1)")

q = st.text_input("Frage / Query")
k = st.slider("Top-k", 1, 20, 5)

if st.button("Suchen") and q:
    hits = search(q, top_k=k)
    for i, h in enumerate(hits, 1):
        with st.expander(f"{i}. {h['doc_id']} • Seite {h['page_start']} • Score {h['score']:.3f}"):
            st.write(h["text"])
