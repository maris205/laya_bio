---
dataset_info:
  features:
  - name: sequence
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 18795028
    num_examples: 45619
  download_size: 8797976
  dataset_size: 18795028
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
