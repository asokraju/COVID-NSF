#!/bin/bash -i

for weight in 0.1 0.2 0.4 0.6 0.8 1.0
do
./bash-scripts/batch.sh $weight
done
