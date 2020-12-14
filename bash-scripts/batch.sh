#!/bin/bash -i

WEIGHT=$1
WEIGHT_DIR="${WEIGHT}"
DIR=$PWD
PARENT_DIR="$(dirname $PWD)"
mkdir -p ./results/$WEIGHT_DIR
cd ./results/$WEIGHT_DIR

conda activate tf-gpu

export run_exec=$DIR/run.py #python script that we want to run
export run_flags="--env_weight=${WEIGHT} --summary_dir=$PWD"   #flags for the script
/home/kosaraju/anaconda3/envs/tf-gpu/bin/python $run_exec $run_flags