#! /bin/bash

input_dim=2
max_outputs=600
init_dim=64
n_mixtures=16
z_dim=16
hidden_dim=64

epochs=200
dataset_type=multimnist
mnist_data_dir="cache/mnist"
multimnist_data_dir="cache/multimnist"

python sample_and_summarize.py \
  --input_dim ${input_dim} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 2 4 8 16 32 \
  --hidden_dim ${hidden_dim} \
  --num_heads 4 \
  --epochs ${epochs} \
  --dataset_type ${dataset_type} \
  --log_name "gen/multimnist/camera-ready" \
  --mnist_data_dir ${mnist_data_dir} \
  --multimnist_data_dir ${multimnist_data_dir} \
  --slot_att \
  --ln \
  --seed 42

echo "Done"
exit 0
