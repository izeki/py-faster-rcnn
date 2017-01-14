#!/bin/bash
# Usage:
# ./experiments/scripts/lstm_rcnn.sh GPU NET DATASET [options args to {train_net,test_lstm_rcnn}.py]
# DATASET is caltech.

sset -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

TRAIN_IMDB="caltech_train"
TEST_IMDB="caltech_test"
PI_DIR="caltech"
ITERS=70000

LOG="experiments/logs/caltech_lstm_rcnn_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/lstm_rcnn/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/lstm_rcnn.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_lstm_rcnn.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/lstm_rcnn/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/lstm_rcnn.yml \
  ${EXTRA_ARGS}