#!/bin/bash 

train_cfg_file="configs/train_sd3m_stage2.yml"
script_file="train/train_stage2.py"

NGPUS_PER_NODE=8
MASTER_ADDR="127.0.0.1"
NNODES=1
NODE_RANK=0

torchrun --nproc_per_node=${NGPUS_PER_NODE} --nnodes=${NNODES} --node-rank=${NODE_RANK} \
  --master-addr=${MASTER_ADDR} --master-port=29501 \
  ${script_file} --train_cfg_file "${train_cfg_file}" 
