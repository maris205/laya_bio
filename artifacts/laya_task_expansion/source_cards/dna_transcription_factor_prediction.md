---
dataset_info:
  features:
  - name: sequence
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 3884714
    num_examples: 34378
  download_size: 1764410
  dataset_size: 3884714
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
