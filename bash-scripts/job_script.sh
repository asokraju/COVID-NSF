#!/bin/bash -i

for gamma in 0.3 0.4 0.5 0.6 0.7
do
./bash-scripts/batch.sh $gamma
done
