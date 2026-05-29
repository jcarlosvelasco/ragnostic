import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="ragnostic", page_icon="🧠")
st.title("ragnostic")

tab_query, tab_admin = st.tabs(["Query", "Admin"])

with tab_query:
    query = st.text_input("Enter your query:", placeholder="e.g., what does the documentation say about...?")
    if st.button("Submit query", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Thinking..."):
                try:
                    resp = httpx.post(
                        f"{API_BASE_URL}/retrieve",
                        json={"query": query},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.success("Answer:")
                    st.markdown(data["answer"])
                    if data.get("context"):
                        with st.expander("View retrieved context"):
                            st.write(data["context"])
                except httpx.RequestError as e:
                    st.error(f"Connection error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

with tab_admin:
    st.subheader("Document Loading")
    if st.button("Load documents (load_docs)"):
        with st.spinner("Loading documents..."):
            try:
                resp = httpx.post(f"{API_BASE_URL}/load_docs", timeout=300)
                resp.raise_for_status()
                st.success("Documents loaded successfully.")
            except httpx.RequestError as e:
                st.error(f"Connection error: {e}")

    st.subheader("Data Cleanup")
    if st.button("Clean data (clean_docs)"):
        with st.spinner("Cleaning data..."):
            try:
                resp = httpx.get(f"{API_BASE_URL}/clean", timeout=300)
                resp.raise_for_status()
                st.success("Data cleaned successfully.")
            except httpx.RequestError as e:
                st.error(f"Connection error: {e}")
