"""Visual demo: type a description or pick an item and browse visually similar products.

streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from outfit_search.catalogue import load_items
from outfit_search.config import ENCODERS, default_paths
from outfit_search.encoders import OpenClipEncoder
from outfit_search.index import EmbeddingIndex

PATHS = default_paths()
st.set_page_config(page_title="Outfit similarity", layout="wide")


@st.cache_resource
def load(encoder: str) -> tuple[EmbeddingIndex, pd.DataFrame, OpenClipEncoder]:
    return (
        EmbeddingIndex.load(PATHS.index(encoder)),
        load_items(PATHS.items),
        OpenClipEncoder(encoder, ENCODERS[encoder]),
    )


with st.sidebar:
    encoder_name = st.radio("Encoder", list(ENCODERS), index=1)
    try:
        index, items, encoder = load(encoder_name)
    except FileNotFoundError:
        st.error("Run `make data embed` first.")
        st.stop()
    gender = st.selectbox("Gender", ["Any", *sorted(items["gender"].dropna().unique())])
    category = st.selectbox("Category", ["Any", *sorted(items["masterCategory"].dropna().unique())])
    k = st.slider("Results", 6, 36, 18, 6)

mask = np.ones(len(items), dtype=bool)
if gender != "Any":
    mask &= (items["gender"] == gender).to_numpy()
if category != "Any":
    mask &= (items["masterCategory"] == category).to_numpy()

st.title("Outfit similarity search")
query = st.text_input("Describe what you are looking for", "navy linen shirt for summer")
anchor = st.session_state.get("anchor")

if anchor is not None:
    position = index.position(anchor)
    row = items.iloc[position]
    left, right = st.columns([1, 5])
    left.image(str(PATHS.image(anchor)), width=120)
    right.markdown(f"**More like:** {row['productDisplayName']}")
    if right.button("Back to text search"):
        st.session_state.pop("anchor")
        st.rerun()
    result = index.search(index.vectors[position], k, exclude=np.array([position]), candidates=mask)
else:
    result = index.search(encoder.encode_texts([query]), k, candidates=mask)

columns = st.columns(6)
for slot, (pos, score) in enumerate(zip(result.positions[0], result.scores[0], strict=True)):
    if not np.isfinite(score):
        continue
    row = items.iloc[int(pos)]
    with columns[slot % 6]:
        st.image(str(PATHS.image(int(row["id"]))), use_container_width=True)
        st.caption(f"{row['productDisplayName']}  \n{row['baseColour']} · {score:.2f}")
        if st.button("More like this", key=f"more-{row['id']}"):
            st.session_state["anchor"] = int(row["id"])
            st.rerun()
