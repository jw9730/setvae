#! /bin/bash

input_dim=2
max_outputs=400
init_dim=32
n_mixtures=4
z_dim=16
hidden_dim=64
num_heads=4

epochs=200
dataset_type=mnist
mnist_data_dir="cache/mnist"

python sample_and_summarize.py \
  --input_dim ${input_dim} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 2 4 8 16 32 \
  --hidden_dim ${hidden_dim} \
  --num_heads ${num_heads} \
  --epochs ${epochs} \
  --dataset_type ${dataset_type} \
  --log_name "gen/mnist/camera-ready" \
  --mnist_data_dir ${mnist_data_dir} \
  --slot_att \
  --ln \
  --seed 42

echo "Done"
exit 0
