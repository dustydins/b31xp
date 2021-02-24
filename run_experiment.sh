#!/usr/bin/env bash

rm results/current_test.csv;

echo 'NUMBER OF NODES: 4';
./run.sh 1  ./main.py -e NUM_NODES -nn 4 -sh;
./run.sh 19 ./main.py -e NUM_NODES -nn 4;
echo 'NUMBER OF NODES: 8';
./run.sh 20 ./main.py -e NUM_NODES -nn 8;
echo 'NUMBER OF NODES: 16';
./run.sh 20 ./main.py -e NUM_NODES -nn 16;
echo 'NUMBER OF NODES: 32';
./run.sh 20 ./main.py -e NUM_NODES -nn 32;
echo 'NUMBER OF NODES: 64';
./run.sh 20 ./main.py -e NUM_NODES -nn 64;
echo 'NUMBER OF NODES: 128';
./run.sh 20 ./main.py -e NUM_NODES -nn 128;

./experiments.py -f num_nodes_linear_mlp -a;
