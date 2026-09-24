# Traditional biological vocabulary → CPT → SFT: first paired round

Final-step results on the existing promoter selection-development split (1,052 examples). One seed. No test inference. Both arms use the same expanded vocabulary, original encoder initialization, classifier initialization, training order and SFT schedule; CPT adds unsupervised adaptation.

| Training rows | Arm | Updates | Train accuracy | Dev accuracy | Dev macro-F1 | Dev NLL |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | no_CPT | 192 | 0.8340 | 0.8251 | 0.8244 | 0.3873 |
| 4096 | CPT | 192 | 0.9062 | 0.8755 | 0.8755 | 0.3033 |
| 16766 | no_CPT | 786 | 0.9232 | 0.9002 | 0.9002 | 0.2511 |
| 16766 | CPT | 786 | 0.9512 | 0.8992 | 0.8992 | 0.2446 |

## Paired changes

- 4096: CPT − no CPT = +5.04 accuracy percentage points; paired group-bootstrap 95% interval [+3.14, +7.03] pp.
- full: CPT − no CPT = -0.10 accuracy percentage points; paired group-bootstrap 95% interval [-1.43, +1.24] pp.

## Masked-language validation

| Step | Phase | DNA NLL | Protein NLL | Text NLL |
|---:|---|---:|---:|---:|
| 0 | initial | 14.4109 | 14.2515 | 12.8064 |
| 128 | warmup | 6.0509 | 6.1106 | 5.9664 |
| 384 | full | 5.6757 | 5.6424 | 3.1054 |
| 640 | full | 5.5694 | 5.5244 | 2.5040 |
| 896 | full | 5.5178 | 5.4718 | 2.2655 |
| 1152 | full | 5.4975 | 5.4491 | 2.1957 |

## Training integrity and limits

- Added tokens: 2044; natural input coverage: 2040; masked-target coverage: 2039.
- Old embedding rows exactly invariant during warmup: True. All final checkpoint prediction reload checks passed.
- A new MLM prediction head was necessary because the starting checkpoint contains only the original encoder and Laya heads. Initial MLM loss reduction includes head training; it is not pure encoder acquisition.
- Fixed three-epoch SFT means the two dataset sizes use different update budgets. The paired CPT effect is assessed separately at each size.
- CPT uses a filtered, length-biased sample from historical corpora. Long-kmer guards do not prove global homology independence. Future tasks require admission checks.
- Development memberships have historical use; uncertainty intervals are descriptive. This single-seed round does not establish task-general, few-shot or zero-shot performance.
- CPT final model and optimizer/RNG/sampler state and both full-data SFT checkpoints are retained. Small-data checkpoints are removed after exact reload verification; their full predictions remain.

Local source artifacts: `/root/autodl-tmp/jev_gene/artifacts/laya_biocpt_v2`.
