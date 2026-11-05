"""Outfit search: describe it, show a photo, or pick a product and see what looks alike.

Run ``make data embed`` once, then ``make app``.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from outfit_search.catalogue import load_items
from outfit_search.config import ENCODERS, default_paths
from outfit_search.encoders import Embeddings, OpenClipEncoder
from outfit_search.index import EmbeddingIndex

PATHS = default_paths()
REPO = "https://github.com/iamkane22/outfit-similarity"
EXAMPLES = (
    "white leather sneakers",
    "red party dress",
    "black leather handbag",
    "navy linen shirt",
    "denim jacket",
    "gold watch",
)
CATEGORIES = ("Apparel", "Footwear", "Accessories")
SWATCH = {
    "Black": "#111111", "White": "#FFFFFF", "Blue": "#2563EB", "Brown": "#7C4A2D",
    "Grey": "#9CA3AF", "Red": "#DC2626", "Green": "#16A34A", "Navy Blue": "#1E2A4A",
    "Pink": "#F9A8D4", "Purple": "#7E22CE", "Silver": "#C0C0C0", "Beige": "#E8DCC4",
    "Yellow": "#FACC15", "Gold": "#C9A227", "Maroon": "#7A1F2E", "Orange": "#F97316",
    "Cream": "#FFFDD0", "Olive": "#708238", "Steel": "#71797E", "Charcoal": "#36454F",
    "Off White": "#F8F5EE", "Peach": "#FFCBA4", "Magenta": "#C026D3", "Skin": "#E8BEAC",
    "Lavender": "#C4B5FD", "Khaki": "#C3B091", "Grey Melange": "#B8B8B8",
    "Mustard": "#E1AD01", "Teal": "#0D9488",
}  # fmt: skip

st.set_page_config(page_title="Outfit search", page_icon="👗", layout="wide")
st.html(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1240px;}
      h1 {font-size: 1.9rem !important; font-weight: 650 !important; letter-spacing: -0.02em;}
      .lede {color: #4B5563; margin: -0.6rem 0 1rem 0;}
      .card {border: 1px solid #EEF0F2; border-radius: 14px; padding: 10px; background: #fff;}
      .thumb {height: 170px; display: flex; align-items: center; justify-content: center;
              background: #FAFAFA; border-radius: 10px; overflow: hidden;}
      .thumb img {height: 158px; width: auto; max-width: 100%; object-fit: contain;}
      .name {font-size: 0.83rem; color: #111827; margin-top: 8px; line-height: 1.25;
             height: 2.5em; overflow: hidden; display: -webkit-box;
             -webkit-line-clamp: 2; -webkit-box-orient: vertical;}
      .meta {font-size: 0.75rem; color: #6B7280; margin-top: 4px;
             display: flex; align-items: center; gap: 6px; white-space: nowrap;}
      .meta .text {overflow: hidden; text-overflow: ellipsis;}
      .dot {width: 10px; height: 10px; border-radius: 50%; border: 1px solid #D1D5DB;
            display: inline-block; flex: none;}
      .banner {display: flex; gap: 16px; align-items: center; padding: 12px;
               border: 1px solid #EEF0F2; border-radius: 14px; background: #FAFAFA;}
      .banner img {height: 90px; border-radius: 8px; background: #fff;}
      .banner .label {font-size: 0.75rem; color: #6B7280; text-transform: uppercase;
                      letter-spacing: 0.06em;}
      .banner .title {font-size: 1rem; color: #111827; font-weight: 600;}
      .note {color: #6B7280; font-size: 0.82rem;}
    </style>
    """
)


@st.cache_resource(show_spinner="Loading the search model (first start takes ~30 s)...")
def load(encoder: str) -> tuple[EmbeddingIndex, pd.DataFrame, OpenClipEncoder]:
    return (
        EmbeddingIndex.load(PATHS.index(encoder)),
        load_items(PATHS.items),
        OpenClipEncoder(encoder, ENCODERS[encoder]),
    )


@st.cache_data(max_entries=5000)
def thumbnail(item_id: int) -> str:
    return base64.b64encode(PATHS.image(item_id).read_bytes()).decode()


def card(row: pd.Series) -> str:
    colour = row["baseColour"] if isinstance(row["baseColour"], str) else "—"
    dot = SWATCH.get(colour, "conic-gradient(#f87171,#facc15,#4ade80,#60a5fa,#f87171)")
    return (
        f'<div class="card"><div class="thumb"><img src="data:image/jpeg;base64,'
        f'{thumbnail(int(row["id"]))}"/></div>'
        f'<div class="name">{row["productDisplayName"]}</div>'
        f'<div class="meta"><span class="dot" style="background:{dot}"></span>'
        f'<span class="text">{colour} · {row["articleType"]}</span></div></div>'
    )


state = st.session_state
state.setdefault("query", EXAMPLES[3])
state.setdefault("anchor", None)
state.setdefault("upload_key", 0)


def use_example() -> None:
    if state.get("example"):
        state["query"] = state["example"]
        state["anchor"] = None


def show_similar(item_id: int) -> Callable[[], None]:
    def callback() -> None:
        state["anchor"] = item_id

    return callback


def reset() -> None:
    state["anchor"] = None
    state["upload_key"] += 1


# -- header and controls ---------------------------------------------------------
title, settings = st.columns([5, 1], vertical_alignment="bottom")
title.title("Outfit search")
with settings.popover("Settings", width="stretch"):
    encoder_name = st.radio(
        "Model",
        list(ENCODERS),
        index=1,
        format_func={"clip": "CLIP (general)", "fashion-clip": "FashionCLIP"}.__getitem__,
        help="FashionCLIP is tuned on fashion data and scores higher on every metric.",
    )
    k = st.slider("Results", 6, 36, 18, 6)

index, items, encoder = load(encoder_name)
st.html(
    f'<p class="lede">Describe what you want, show a photo, or open any product to see '
    f"what looks alike. {len(items):,} products, FashionCLIP embeddings.</p>"
)

search, photo = st.columns([5, 1], vertical_alignment="bottom")
search.text_input(
    "Search",
    key="query",
    placeholder="e.g. white leather sneakers",
    label_visibility="collapsed",
    on_change=lambda: state.update(anchor=None),
)
with photo.popover("Search by photo", width="stretch"):
    upload = st.file_uploader(
        "Photo", type=["jpg", "jpeg", "png", "webp"], key=f"upload-{state['upload_key']}"
    )

st.pills("Try", EXAMPLES, key="example", on_change=use_example, label_visibility="collapsed")

filters = st.columns([2, 3])
gender = filters[0].segmented_control(
    "Gender", ["Men", "Women", "Unisex", "Kids"], selection_mode="single"
)
categories = filters[1].pills("Category", CATEGORIES, selection_mode="multi")

# The catalogue files some kidswear under Men/Women; the product name is more reliable.
is_kids = (
    items["gender"].isin(["Boys", "Girls"])
    | items["productDisplayName"].str.contains(
        r"Kidswear|Girl's|Boy's|Infant", case=False, regex=True, na=False
    )
).to_numpy()
mask = np.ones(len(items), dtype=bool)
if gender == "Kids":
    mask &= is_kids
elif gender:
    mask &= (items["gender"] == gender).to_numpy() & ~is_kids
if categories:
    mask &= items["masterCategory"].isin(categories).to_numpy()

# -- query -------------------------------------------------------------------------
exclude: np.ndarray | None = None
query: Embeddings
if state["anchor"] is not None and int(state["anchor"]) in set(index.ids.tolist()):
    position = index.position(state["anchor"])
    anchor_row = items.iloc[position]
    query, exclude = index.vectors[position], np.array([position])
    banner = (
        f'<div class="banner"><img src="data:image/jpeg;base64,'
        f'{thumbnail(int(anchor_row["id"]))}"/>'
        f'<div><div class="label">More like this</div>'
        f'<div class="title">{anchor_row["productDisplayName"]}</div></div></div>'
    )
elif upload is not None:
    image = Image.open(upload).convert("RGB")
    query = encoder.encode_images([image])
    thumb = image.copy()
    thumb.thumbnail((180, 180))
    buffer = BytesIO()
    thumb.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    banner = (
        f'<div class="banner"><img src="data:image/jpeg;base64,{encoded}"/>'
        '<div><div class="label">Search by photo</div>'
        '<div class="title">Products that look like your photo</div></div></div>'
    )
else:
    query = encoder.encode_texts([state["query"] or EXAMPLES[3]])
    banner = ""

if banner:
    left, right = st.columns([5, 1], vertical_alignment="center")
    left.html(banner)
    right.button("Back to search", on_click=reset, width="stretch")

# -- results -----------------------------------------------------------------------
result = index.search(query, k, exclude=exclude, candidates=mask)
hits = [
    int(p) for p, s in zip(result.positions[0], result.scores[0], strict=True) if np.isfinite(s)
]
if not hits:
    st.info("No products match these filters.")
for start in range(0, len(hits), 6):
    for column, position in zip(st.columns(6), hits[start : start + 6], strict=False):
        row = items.iloc[position]
        column.html(card(row))
        column.button(
            "Similar items →",
            key=f"similar-{row['id']}",
            on_click=show_similar(int(row["id"])),
            type="tertiary",
        )

st.html(
    f'<p class="note">Catalogue: Fashion Product Images (Small), MIT licence. '
    f'Code and evaluation: <a href="{REPO}">{REPO.removeprefix("https://")}</a></p>'
)
