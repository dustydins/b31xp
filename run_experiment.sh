#!/usr/bin/env bash

rm results/current_test.csv;

echo 'NUMBER OF HIDDEN LAYERS: 1';
./run.sh 1  ./main.py -e NUM_HIDDEN -nh 1 -sh;
./run.sh 19 ./main.py -e NUM_HIDDEN -nh 1;
echo 'NUMBER OF HIDDEN LAYERS: 2';
./run.sh 20 ./main.py -e NUM_HIDDEN -nh 2;
echo 'NUMBER OF HIDDEN LAYERS: 3';
./run.sh 20 ./main.py -e NUM_HIDDEN -nh 3;
echo 'NUMBER OF HIDDEN LAYERS: 4';
./run.sh 20 ./main.py -e NUM_HIDDEN -nh 4;

./experiments.py -f num_hidden_linear_mlp -a;
