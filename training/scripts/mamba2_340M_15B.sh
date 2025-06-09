#!/usr/bin/env bash

###############################################################################
# 将本文件保存为 run_train.sh 后，执行以下命令即可运行：
#   chmod +x run_train.sh
#   ./run_train.sh
###############################################################################

bash train.sh \
  type=mamba2 \
  lr=3e-4 \
  scheduler=cosine_with_min_lr \
  batch=32 \
  update=1 \
  warmup=1024 \
  steps=30720 \
  context=2048 \
  gpus=8 \
  nodes=1 \
  path=exp/mamba2_340M_15B\
  logging=32 \
  seed=42 \
  project=fla-Slim \
  model=configs/mamba2_340M.json \
  data=SlimPajama \
  cache=/cpfs04/shared/MOE/datasets/fla-15B/SlimPajama/train
