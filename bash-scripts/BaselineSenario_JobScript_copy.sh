#!/bin/bash -i
for test_name in BaselineSenario
do
  for random_seed in 1321 456 112
  do
    for max_episodes in 20
    do
      for epsilon in 0.02 0.05 0.1
      do
        for weight in 0.5 0.6
        do
        ./bash-scripts/BaselineSenario_Batch.sh $test_name $random_seed $max_episodes $epsilon $weight
        done
      done
    done
  done
done

