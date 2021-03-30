#! /bin/bash

input_dim=2
max_outputs=600
init_dim=64
n_mixtures=16
z_dim=16
hidden_dim=64
num_heads=4

lr=1e-3
beta=1e-2
epochs=200
kl_warmup_epochs=40
scheduler="linear"
dataset_type=multimnist
log_name=gen/multimnist/camera-ready
mnist_data_dir="cache/mnist"
multimnist_data_dir="cache/multimnist"

deepspeed train.py \
  --kl_warmup_epochs ${kl_warmup_epochs} \
  --input_dim ${input_dim} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 2 4 8 16 32 \
  --hidden_dim ${hidden_dim} \
  --num_heads ${num_heads} \
  --lr ${lr} \
  --beta ${beta} \
  --epochs ${epochs} \
  --dataset_type ${dataset_type} \
  --log_name ${log_name} \
  --mnist_data_dir ${mnist_data_dir} \
  --multimnist_data_dir ${multimnist_data_dir} \
  --resume_optimizer \
  --save_freq 10 \
  --viz_freq 100 \
  --log_freq 10 \
  --val_freq 1000 \
  --scheduler ${scheduler} \
  --slot_att \
  --ln \
  --seed 42 \
  --distributed \
  --deepspeed_config batch_size.json

echo "Done"
exit 0
