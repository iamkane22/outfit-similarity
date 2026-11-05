"""Side-by-side comparison of two encoders: metrics table, significance and examples."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from outfit_search.config import Paths
from outfit_search.evaluation import chance_precision, paired_bootstrap, relevance
from outfit_search.index import EmbeddingIndex

logger = logging.getLogger(__name__)

ROWS: dict[str, str] = {
    "item2item_precision@10": "Similar items: precision@10 (same type and colour)",
    "item2item_hit@1": "Similar items: top result relevant (hit@1)",
    "item2item_map@10": "Similar items: mAP@10",
    "zeroshot_type_accuracy": "Zero-shot article type, 41 classes: accuracy",
    "zeroshot_type_macro_f1": "Zero-shot article type: macro-F1",
    "zeroshot_colour_accuracy": "Zero-shot colour, 22 classes: accuracy",
    "text2image_recall@1": "Product title -> image: recall@1",
    "text2image_recall@10": "Product title -> image: recall@10",
    "text2image_mrr": "Product title -> image: MRR",
}


def comparison_markdown(paths: Paths, baseline: str, candidate: str, n_queries: int = 2000) -> str:
    metrics = {
        name: json.loads((paths.reports_dir / f"metrics_{name}.json").read_text())
        for name in (baseline, candidate)
    }
    lines = [f"| Task | {baseline} | {candidate} |", "|---|---:|---:|"]
    for key, label in ROWS.items():
        lines.append(f"| {label} | {metrics[baseline][key]:.1%} | {metrics[candidate][key]:.1%} |")

    from outfit_search.catalogue import load_items

    items = load_items(paths.items)
    queries = np.random.default_rng(0).choice(
        len(items), size=min(n_queries, len(items)), replace=False
    )
    per_query = {
        name: relevance(EmbeddingIndex.load(paths.index(name)), items, queries).mean(axis=1)
        for name in (baseline, candidate)
    }
    mean, low, high = paired_bootstrap(per_query[candidate], per_query[baseline])
    lines += [
        "",
        f"Random ranking would give precision@10 of {chance_precision(items):.1%}.",
        f"{candidate} minus {baseline}, precision@10: {mean:+.1%} "
        f"(95% paired bootstrap interval {low:+.1%} to {high:+.1%}, {len(queries)} queries).",
    ]
    return "\n".join(lines) + "\n"


def example_grid(
    paths: Paths,
    encoders: tuple[str, ...],
    query_ids: list[int],
    k: int = 5,
    target: Path | None = None,
) -> Path:
    """Query image followed by the top-``k`` neighbours for each encoder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from outfit_search.catalogue import load_items

    items = load_items(paths.items)
    indexes = {name: EmbeddingIndex.load(paths.index(name)) for name in encoders}
    rows = [(qid, name) for qid in query_ids for name in encoders]
    fig, axes = plt.subplots(len(rows), k + 1, figsize=(1.6 * (k + 1), 2.05 * len(rows)))
    for r, (qid, name) in enumerate(rows):
        index = indexes[name]
        position = index.position(qid)
        result = index.search(index.vectors[position], k, exclude=np.array([position]))
        query_row = items.iloc[position]
        shown = [(qid, f"query ({name})\n{query_row['articleType']}, {query_row['baseColour']}")]
        for pos in result.positions[0]:
            row = items.iloc[int(pos)]
            match = (row["articleType"], row["baseColour"]) == (
                query_row["articleType"],
                query_row["baseColour"],
            )
            mark = "✓" if match else "✗"
            shown.append((int(row["id"]), f"{mark} {row['articleType']}\n{row['baseColour']}"))
        for c, (item_id, caption) in enumerate(shown):
            ax = axes[r, c]
            with Image.open(paths.image(item_id)) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_title(caption, fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    target = target or paths.reports_dir / "figures" / "examples.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130)
    plt.close(fig)
    logger.info("Wrote %s", target)
    return target
