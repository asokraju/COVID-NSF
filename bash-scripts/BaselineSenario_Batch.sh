#!/bin/bash -i
test_name=$1
random_seed=$2
max_episodes=$3
epsilon=$4
weight=$5

# creating a directory structure to save the results
CURRENT_DIR=$PWD
PARENT_DIR="$(dirname $PWD)"

TEST_NAME_DIR="test_name=${TEST_NAME}"
RANDOM_SEED_DIR="seed=${random_seed}"
MAX_EPISODES_DIR="max_episodes=${max_episodes}"
EPSILON_DIR="epsilon=${epsilon}"
WEIGHT_DIR="w=${weight}"

mkdir -p $TEST_NAME_DIR                    # making a directory with test name
RESULTS_DIR=${PARENT_DIR}/${TEST_NAME_DIR} # Directory for results
cd $RESULTS_DIR                            # we are inside the results_dir

mkdir -p $RANDOM_SEED_DIR
cd $RANDOM_SEED_DIR

mkdir -p $MAX_EPISODES_DIR
cd $MAX_EPISODES_DIR

mkdir -p $EPSILON_DIR
cd $EPSILON_DIR

mkdir -p $WEIGHT_DIR
cd $WEIGHT_DIR


# python script that we want to run
export run_exec=$PARENT_DIR/run.py


# flags for the script
export run_flags="--test_name=${test_name} --random_seed=${random_seed} --max_episodes=$max_episodes --epsilon=${epsilon} --weight=${weight} --summary_dir='$PWD'" 

conda activate tf-gpu
/home/kosaraju/anaconda3/envs/tf-gpu/bin/python $run_exec $run_flags