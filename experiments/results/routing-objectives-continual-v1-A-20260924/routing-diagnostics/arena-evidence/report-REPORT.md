# RouterArena difficulty-label / action-slot alignment

Frozen λ=0 diagnostic; 512 tokens; single seed 42. R0 predicts difficulty; R1/R2 project candidate action slots and are **not difficulty classifiers**. These results do not select a new winner, estimate Claude quality/cost, provide an official Arena Score, or approve deployment.

Full and robustness are paired overlapping data; their counts must not be added as independent samples. Historical RouterArena observations are disclosed. Near-duplicate detection is approximate; consult overlap-audit.json.

| Method | Slice | n | Alignment accuracy | Macro-F1 (3 labels) | Balanced accuracy | QWK | Hard→easy |
|---|---|---:|---:|---:|---:|---:|---:|
| ordinal | full_clean | 8379 | 0.4548 | 0.3961 | 0.4062 | 0.1340 | 0.5234 |
| ordinal | rb_service_aligned | 7351 | 0.4771 | 0.4045 | 0.4055 | 0.1791 | 0.4882 |
| ordinal | rb_unseen | 1028 | 0.2957 | 0.3067 | 0.4758 | -0.0451 | 0.5887 |
| pointwise | full_clean | 8379 | 0.2280 | 0.1754 | 0.3110 | -0.0357 | 0.0423 |
| pointwise | rb_service_aligned | 7351 | 0.1875 | 0.1495 | 0.3293 | -0.0128 | 0.0392 |
| pointwise | rb_unseen | 1028 | 0.5175 | 0.3174 | 0.3480 | -0.0103 | 0.0480 |
| pairwise | full_clean | 8379 | 0.2136 | 0.1927 | 0.2744 | -0.0520 | 0.0988 |
| pairwise | rb_service_aligned | 7351 | 0.1981 | 0.1839 | 0.3083 | -0.0066 | 0.0917 |
| pairwise | rb_unseen | 1028 | 0.3249 | 0.2277 | 0.2690 | -0.1522 | 0.1119 |

Raw scores and token/truncation data remain in the sealed inference artifact. Per-question projections are in predictions.parquet; category/domain/source metrics are in category-domain-source-metrics.csv. Both directions of the conditional confusion table, label and predicted support, and paired robustness are in summary.json. Missing strata or undefined statistics are N/A, not zero performance.
