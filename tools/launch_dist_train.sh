#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

source /mnt/vepfs/Perception/perception-users/hongliang/condanew/anaconda3/bin/activate \
  /mnt/vepfs/Perception/perception-users/hongliang/condanew/anaconda3/envs/sparse_nuscenes


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# PYTHONPATH="/root/code/MonoCon/mmdetection3d-0.14.0":"/root/code/MonoCon/mmdetection-2.11.0":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# PYTHONPATH="/root/code/mmdetection3d-1.0.0.dev0":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
PYTHONPATH="/root/code/Sparse4D-nuscenes/mmdetection3d":$PYTHONPATH \
OMP_NUM_THREADS=4 python3 -m torch.distributed.launch --nproc_per_node "$MLP_WORKER_GPU" \
  --master_addr "$MLP_WORKER_0_HOST" \
  --master_port "$MLP_WORKER_0_PORT" \
  --node_rank "$MLP_ROLE_INDEX" \
  --nnodes="$MLP_WORKER_NUM" $(dirname "$0")/train.py \
  --launcher=pytorch "${@:1}"

