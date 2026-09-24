# Six-task engineering pilot artifacts

Two completed 100-update train/dev runs, plus a launch record for the subsequent three-pass development round. These are not a new held-out paper benchmark.

- `data_manifest.json`: fixed source/output hashes, split guards, exclusions, score anchors and limitations.
- `additional_sources.json`: DeepLoc 2.0 CSV and TAPE fluorescence archive URLs and content hashes.
- `provenance.json`: original Laya weight fingerprint and source code fingerprints.
- `candidate/`, `shared_heads/`: completed run configs, loss traces and summaries. Checkpoints and per-example files remain in the local workspace.
- `comparison.json`: full before/after metrics, matched entity exposure check, training-only constants and alignment baseline.
- `development_round1_launch.json`: static launch record, not live status or completed results.

Interpretation, sources, caveats and local live-status location: [execution report](../../research/laya_multitask_pilot.md).
