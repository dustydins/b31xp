#!/usr/bin/env bash

rm ../results/current_test.csv;

# subsequence

# echo 'SUBSEQUENCE: 1';
# ./run.sh 1  ./main.py -a 32 32 -e SUBSEQUENCE_SIZE -ss 1 -sh;
# ./run.sh 19 ./main.py -a 32 32 -e SUBSEQUENCE_SIZE -ss 1;
# echo 'SUBSEQUENCE: 2';
# ./run.sh 20 ./main.py -a 32 32 -e  SUBSEQUENCE_SIZE -ss 2;
# echo 'SUBSEQUENCE: 4';
# ./run.sh 20 ./main.py -a 32 32 -e  SUBSEQUENCE_SIZE -ss 4;
# echo 'SUBSEQUENCE: 6';
# ./run.sh 20 ./main.py -a 32 32 -e  SUBSEQUENCE_SIZE -ss 6;
# echo 'SUBSEQUENCE: 8';
# ./run.sh 20 ./main.py -a 32 32 -e  SUBSEQUENCE_SIZE -ss 8;
# echo 'SUBSEQUENCE: 10';
# ./run.sh 20 ./main.py -a 32 32 -e  SUBSEQUENCE_SIZE -ss 10;
# ./experiments.py -f subsequence_mlp -a;

# num nodes

echo 'NUM NODES: 2';
./run.sh 1  ./main.py -a 2 2 -e ARCHITECTURE -ss 4 -sh;
./run.sh 19 ./main.py -a 2 2 -e ARCHITECTURE -ss 4;
echo 'NUM NODES: 4';
./run.sh 20 ./main.py -a 4 4 -e  ARCHITECTURE -ss 4;
echo 'NUM NODES: 8';
./run.sh 20 ./main.py -a 8 8 -e  ARCHITECTURE -ss 4;
echo 'NUM NODES: 16';
./run.sh 20 ./main.py -a 16 16 -e  ARCHITECTURE -ss 4;
echo 'NUM NODES: 32';
./run.sh 20 ./main.py -a 32 32 -e  ARCHITECTURE -ss 4;
./experiments.py -f num_nodes_mlp -a;
