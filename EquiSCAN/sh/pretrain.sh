#!/bin/bash

## Help:
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0`
    1: LAYER_TYPE   GRU LSTM RNN
    2: CUDA_DEVICE  0 1 2 3
    3: MODE         test train
    "
  exit 0
fi

LAYER_TYPE=$1
CUDA_DEVICE=$2
MODE=$3
n_iters=200000

cd ../
if [ "$MODE" == "train" ]; then
  SPLIT=turn_up_jump_turn_left
  for SEED in 0 1 2; do
    echo "TRAIN seed ${SEED} split ${SPLIT} layer_type ${LAYER_TYPE}"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_seq2seq_scan.py --layer_type $LAYER_TYPE --use_attention --bidirectional --split $SPLIT \
    --n_iters $n_iters --save_dir ./saved_models/pretrained_models/ --expanded --seed $SEED
  done

elif [ "$MODE" == "test" ]; then
  for SPLIT in jump turn_left turn_up turn_up_jump_turn_left ; do
    for SEED in 0 1 2; do
      echo "TEST seed ${SEED} split ${SPLIT} layer_type ${LAYER_TYPE}"
      CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python test_seq2seq_scan.py \
      --experiment_dir ./saved_models/pretrained_models/rnn_${LAYER_TYPE}_hidden_64_directions_2/seed_${SEED}/model0/ \
      --best_validation --split $SPLIT --compute_test_accuracy
    done
  done
fi