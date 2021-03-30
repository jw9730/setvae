#! /bin/bash

root=$(pwd)

# Compile CUDA kernel for CD/EMD loss
cd metrics/pytorch_structural_losses/ || exit
make clean
make
cd "$root" || exit
