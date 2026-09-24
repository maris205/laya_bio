---
dataset_info:
  features:
  - name: sequence
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 4854072
    num_examples: 59196
  download_size: 2197851
  dataset_size: 4854072
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
