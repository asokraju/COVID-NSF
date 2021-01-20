#!/bin/bash -i

WEIGHT=$1
WEIGHT_DIR="${WEIGHT}"
DIR=$PWD
PARENT_DIR="$(dirname $PWD)"
<<<<<<< HEAD
mkdir -p ./results/Senario-2/$WEIGHT_DIR
cd ./results/Senario-2/$WEIGHT_DIR
=======
mkdir -p ./results/Senario-1/$WEIGHT_DIR
cd ./results/Senario-1/$WEIGHT_DIR
>>>>>>> 36b895b739fa40919ab1616cb5c07d84657152ab

conda activate tf-gpu

export run_exec=$DIR/run.py #python script that we want to run
export run_flags="--env_weight=${WEIGHT} --summary_dir=$PWD"   #flags for the script
/home/kosaraju/anaconda3/envs/tf-gpu/bin/python $run_exec $run_flags