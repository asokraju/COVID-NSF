#!/bin/bash -i

for weight in 1.0 0.7 0.9 0.8 0.1
do
./bash-scripts/batch.sh $weight
done
