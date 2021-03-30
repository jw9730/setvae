#! /bin/bash

input_dim=3
max_outputs=2500
init_dim=32
n_mixtures=4
z_dim=16
hidden_dim=64
num_heads=4

epochs=8000
dataset_type=shapenet15k
log_name=gen/shapenet15k-airplane/camera-ready
shapenet_data_dir="/data4/ShapeNet/ShapeNetCore.v2.PC15k"

python sample_and_summarize.py \
  --cates airplane \
  --input_dim ${input_dim} \
  --max_outputs ${max_outputs} \
  --init_dim ${init_dim} \
  --n_mixtures ${n_mixtures} \
  --z_dim ${z_dim} \
  --z_scales 1 1 2 4 8 16 32 \
  --hidden_dim ${hidden_dim} \
  --num_heads ${num_heads} \
  --fixed_gmm \
  --train_gmm \
  --epochs ${epochs} \
  --dataset_type ${dataset_type} \
  --log_name ${log_name} \
  --shapenet_data_dir ${shapenet_data_dir} \
  --slot_att \
  --ln \
  --eval \
  --seed 42

echo "Done"
exit 0
