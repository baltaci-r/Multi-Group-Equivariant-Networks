#!/bin/bash

## Help:
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0`
    1: LAYER_TYPE   GRU LSTM RNN
    2: CUDA_DEVICE  0 1 2 3
    3: CONFIG       multi_equi  equi
    "
  exit 0
fi

cd ../

LAYER_TYPE=$1
CUDA_DEVICE=$2
CONFIG=$3

SPLIT=turn_up_jump_turn_left
for SPLIT in turn_left turn_up jump turn_up_jump_turn_left; do
  for SEED in 0 1 2; do
    echo "TEST seed ${SEED} split ${SPLIT} layer_type ${LAYER_TYPE} CONFIG ${CONFIG}"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python test.py --split $SPLIT --expanded \
      --load_model_path "./saved_models/${CONFIG}tuned_models/rnn_${LAYER_TYPE}_hidden_64_directions_2/seed_${SEED}/model0/" \

  done
done

#jupyter_container --command="bash finetune.sh GRU 0 train equi"
#jupyter_container --command="bash finetune.sh LSTM 1 train equi"
#jupyter_container --command="bash finetune.sh RNN 2 train equi"

#jupyter_container --command="bash finetune.sh GRU 0 train multi_equi"
#jupyter_container --command="bash finetune.sh LSTM 1 train multi_equi"
#jupyter_container --command="bash finetune.sh RNN 2 train multi_equi"