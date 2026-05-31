#!/usr/bin/env bash
set -euo pipefail

# --- Edit these parameters as needed ---
dataset_name="SYSU"
device="cuda:5"
num_epoch="10"
lr="5e-6"
root_dir="/data0/zza_data/reid/tireid_data/converted/ItR"
batch_size="96"
seed="2"
optimizer="Adam"
weight_decay="4e-5"
momentum="0.9"
num_workers="8"
text_length="77"
vocab_size="49408"
warmup_factor="0.1"
lrscheduler="cosine"
num_instance="4"
stride_size="16"
output_dir="logs"
log_period="100"
eval_period="1"
extra_args=""
# --------------------------------------

echo "Running: python3 main.py --dataset_name $dataset_name --device $device --num_epoch $num_epoch --lr $lr --batch_size $batch_size --optimizer $optimizer --weight_decay $weight_decay --momentum $momentum --num_workers $num_workers --text_length $text_length --vocab_size $vocab_size --warmup_factor $warmup_factor --lrscheduler $lrscheduler --num_instance $num_instance --stride_size $stride_size --output_dir $output_dir --log_period $log_period --eval_period $eval_period --root_dir $root_dir $extra_args"

python3 main.py --dataset_name "$dataset_name" --device "$device" --seed "$seed" --num_epoch "$num_epoch" \
	--lr "$lr" --batch_size "$batch_size" --optimizer "$optimizer" --weight_decay "$weight_decay" \
	--momentum "$momentum" --num_workers "$num_workers" --text_length "$text_length" --vocab_size "$vocab_size" \
	--warmup_factor "$warmup_factor" --lrscheduler "$lrscheduler" --num_instance "$num_instance" --stride_size "$stride_size" \
	--output_dir "$output_dir" --log_period "$log_period" --eval_period "$eval_period" --root_dir "$root_dir" $extra_args
