| Task | clip | fashion-clip |
|---|---:|---:|
| Similar items: precision@10 (same type and colour) | 30.1% | 33.0% |
| Similar items: top result relevant (hit@1) | 40.2% | 42.5% |
| Similar items: mAP@10 | 22.5% | 25.3% |
| Zero-shot article type, 41 classes: accuracy | 71.6% | 74.3% |
| Zero-shot article type: macro-F1 | 69.1% | 72.4% |
| Zero-shot colour, 22 classes: accuracy | 53.3% | 54.6% |
| Product title -> image: recall@1 | 11.8% | 13.1% |
| Product title -> image: recall@10 | 42.3% | 44.1% |
| Product title -> image: MRR | 21.8% | 23.4% |

Random ranking would give precision@10 of 0.7%.
fashion-clip minus clip, precision@10: +2.9% (95% paired bootstrap interval +2.3% to +3.5%, 2000 queries).
