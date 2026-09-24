---
dataset_info:
- config_name: dna_protein_pair
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 4448263
    num_examples: 4000
  download_size: 2594650
  dataset_size: 4448263
- config_name: dna_protein_pair_rand
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 32269245
    num_examples: 16000
  download_size: 18894709
  dataset_size: 32269245
- config_name: dna_protein_pair_rand_v2
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 29319338
    num_examples: 16000
  download_size: 17532868
  dataset_size: 29319338
- config_name: dna_sim_pair_150bp
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 6257343
    num_examples: 20000
  download_size: 2933986
  dataset_size: 6257343
- config_name: dna_sim_pair_50bp
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 2301109
    num_examples: 20000
  download_size: 1065972
  dataset_size: 2301109
- config_name: dna_sim_pair_simple_150bp
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 5264080
    num_examples: 18000
  download_size: 2443680
  dataset_size: 5264080
- config_name: protein_sim_pair_150bp
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 2081990
    num_examples: 18000
  download_size: 1847847
  dataset_size: 2081990
- config_name: protein_sim_pair_450bp
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 5462986
    num_examples: 18000
  download_size: 5205983
  dataset_size: 5462986
configs:
- config_name: dna_protein_pair
  data_files:
  - split: train
    path: dna_protein_pair/train-*
- config_name: dna_protein_pair_rand
  data_files:
  - split: train
    path: dna_protein_pair_rand/train-*
- config_name: dna_protein_pair_rand_v2
  data_files:
  - split: train
    path: dna_protein_pair_rand_v2/train-*
- config_name: dna_sim_pair_150bp
  data_files:
  - split: train
    path: dna_sim_pair_150bp/train-*
- config_name: dna_sim_pair_50bp
  data_files:
  - split: train
    path: dna_sim_pair_50bp/train-*
- config_name: dna_sim_pair_simple_150bp
  data_files:
  - split: train
    path: dna_sim_pair_simple_150bp/train-*
- config_name: protein_sim_pair_150bp
  data_files:
  - split: train
    path: protein_sim_pair_150bp/train-*
- config_name: protein_sim_pair_450bp
  data_files:
  - split: train
    path: protein_sim_pair_450bp/train-*
---
