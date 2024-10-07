#!/bin/bash

## Help:
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0`
    1: LAYER_TYPE   GRU LSTM RNN
    2: CUDA_DEVICE  0 1 2 3
    3: CONFIG       multi_equi equi canonical
    "
  exit 0
fi

cd ../

LAYER_TYPE=$1
CUDA_DEVICE=$2
CONFIG=$3

mkdir -p ./saved_models/${CONFIG}tuned_models/

SPLIT=turn_up_jump_turn_left
for SEED in 0 1 2; do
  echo "TRAIN seed ${SEED} split ${SPLIT} layer_type ${LAYER_TYPE}"

  echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python equitune.py \
  --config $CONFIG --split $SPLIT --expanded --cuda $CUDA_DEVICE \
  --equivariance \"verb\" --equivariance \"direction-ud\" --equivariance \"direction-rl\" \
  --layer_type $LAYER_TYPE --use_attention --bidirectional --seed $SEED \
  --n_iters 10000 --learning_rate 0.00002 --validation_size 0.01 \
  --save_dir \"./saved_models/${CONFIG}tuned_models/\"
  --load_model_path \"./saved_models/pretrained_models/rnn_${LAYER_TYPE}_hidden_64_directions_2/seed_${SEED}/model0\""

  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python equitune.py \
  --config $CONFIG --split $SPLIT --expanded --cuda $CUDA_DEVICE \
  --equivariance "verb" --equivariance "direction-ud" --equivariance "direction-rl" \
  --layer_type $LAYER_TYPE --use_attention --bidirectional --seed $SEED \
  --n_iters 10000 --learning_rate 0.00002 --validation_size 0.01 \
  --save_dir "./saved_models/${CONFIG}tuned_models/" \
  --load_model_path "./saved_models/pretrained_models/rnn_${LAYER_TYPE}_hidden_64_directions_2/seed_${SEED}/model0"

done

#jupyter_container --command="bash finetune.sh GRU 0 train equi"
#jupyter_container --command="bash finetune.sh LSTM 1 train equi"
#jupyter_container --command="bash finetune.sh RNN 2 train equi"

#jupyter_container --command="bash finetune.sh GRU 0 train multi_equi"
#jupyter_container --command="bash finetune.sh LSTM 1 train multi_equi"
#jupyter_container --command="bash finetune.sh RNN 2 train multi_equi"