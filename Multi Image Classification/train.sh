get_groups () {
  input=$1
  out=""
  for (( i=0; i<${#input}; i++ )); do
    char="${input:$i:1}"
    if   [ $char == "H" ]; then
      out="$out hflip"
    elif  [ $char == "R" ]; then
      out="$out rot90"
    elif  [ $char == "I" ]; then
      out="$out id"
    fi
  done
  echo $out
}

get_fn (){
  if $fusion ; then
      f=T
  else
      f=F
  fi
  file_dir="runs/$dataset/$model/k${K}_fs${f}_g${group}_at${train_augs}_as${test_augs}_in${input_size}_bs${BS}_lr${LR}_ep${EP}_s${SEED}"
}


## Help:
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0`
    1: DATASET  15Scene Caltech101
    "
  exit 0
fi

dataset=$1

BS=64
LR=0.01
EP=100
model=Gemini
input_size=64
#input_size=32


for K in 2 3 4; do
for fusion in true false; do
  a1=$(printf -- 'R%.0s' $(seq 1 $K)) # R*K
  a2=$(printf -- 'I%.0s' $(seq 1 $K)) # I*K
  declare -a groups=("$a1 $a2")

  for group in ${groups[@]}; do
  for train_augs in ${groups[@]}; do
  for test_augs in ${groups[@]}; do

  list_symmetry_groups=$(get_groups $group)
  train_aug=$(get_groups $train_augs)
  test_aug=$(get_groups $test_augs)

  echo DATASET $dataset K $K FUSION $fusion SYMMETRY $list_symmetry_groups TRAIN $train_aug TEST $test_aug

  for SEED in 1 2 3; do

    get_fn
    mkdir -p $file_dir
    echo $file_dir

    cmd="--num_workers 4 --batch_size $BS --lr $LR --num_epochs $EP \
        --dataset $dataset --model_name $model --seed $SEED \
        --conv_channels 16 --lin_channels 64 \
        --train_data_aug_list $train_aug --test_data_aug_list $test_aug  \
        --list_symmetry_groups $list_symmetry_groups \
        --file_dir $file_dir --num_inputs $K --input_size $input_size"

    if $fusion; then
      cmd="$cmd --fusion"
    fi

    if [[ ! -f "$file_dir/results.json" ]] ;
    then
      python main.py $cmd 2>&1 | tee  $file_dir/log.log
    else
      echo "Experiment already exists"
    fi

done
done
done
done
done
echo "***********"
done
