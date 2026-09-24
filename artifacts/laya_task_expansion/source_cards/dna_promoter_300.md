---
dataset_info:
  features:
  - name: sequence
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 18468840
    num_examples: 59195
  download_size: 8661645
  dataset_size: 18468840
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
