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

# echo 'NUM NODES: 2';
# ./run.sh 1  ./main.py -a 2 2 -e ARCHITECTURE -ss 4 -sh;
# ./run.sh 19 ./main.py -a 2 2 -e ARCHITECTURE -ss 4;
# echo 'NUM NODES: 4';
# ./run.sh 20 ./main.py -a 4 4 -e  ARCHITECTURE -ss 4;
# echo 'NUM NODES: 8';
# ./run.sh 20 ./main.py -a 8 8 -e  ARCHITECTURE -ss 4;
# echo 'NUM NODES: 16';
# ./run.sh 20 ./main.py -a 16 16 -e  ARCHITECTURE -ss 4;
# echo 'NUM NODES: 32';
# ./run.sh 20 ./main.py -a 32 32 -e  ARCHITECTURE -ss 4;
# ./experiments.py -f num_nodes_mlp -a;

# data set 

# echo 'DATA SET: 1';
# ./run.sh 1  ./main.py -df 1 -a 8 8 -e DATA_FILE -ss 4 -sh;
# ./run.sh 19 ./main.py -df 1 -a 8 8 -e DATA_FILE -ss 4;
# echo 'DATA SET : 2';
# ./run.sh 20 ./main.py -df 2 -a 8 8 -e  DATA_FILE -ss 4;
# echo 'DATA SET : 3';
# ./run.sh 20 ./main.py -df 3 -a 8 8 -e  DATA_FILE -ss 4;
# echo 'DATA SET : 4';
# ./run.sh 20 ./main.py -df 4 -a 8 8 -e  DATA_FILE -ss 4;
# echo 'DATA SET : 5';
# ./run.sh 20 ./main.py -df 5 -a 8 8 -e  DATA_FILE -ss 4;
# echo 'DATA SET : 6';
# ./run.sh 20 ./main.py -df 6 -a 8 8 -e  DATA_FILE -ss 4;
# echo 'DATA SET : 7';
# ./run.sh 20 ./main.py -df 7 -a 8 8 -e  DATA_FILE -ss 4;
# echo 'DATA SET : 8';
# ./run.sh 20 ./main.py -df 8 -a 8 8 -e  DATA_FILE -ss 4;
# ./experiments.py -f data_set_experiment -a;

# ARCHITECTURE DATA SET 3

# echo 'NUM NODES: 2';
# ./run.sh 1  ./main.py -df 3 -a 2 2 -e ARCHITECTURE -ss 4 -sh;
# ./run.sh 19 ./main.py -df 3 -a 2 2 -e ARCHITECTURE -ss 4;
# echo 'NUM NODES: 4';
# ./run.sh 20 ./main.py -df 3 -a 4 4 -e  ARCHITECTURE -ss 4;
# echo 'NUM NODES: 8';
# ./run.sh 20 ./main.py -df 3 -a 8 8 -e  ARCHITECTURE -ss 4;
# echo 'NUM NODES: 16';
# ./run.sh 20 ./main.py -df 3 -a 16 16 -e  ARCHITECTURE -ss 4;
# echo 'NUM NODES: 32';
# ./run.sh 20 ./main.py -df 3 -a 32 32 -e  ARCHITECTURE -ss 4;
# ./experiments.py -f num_nodes_data_3 -a;

# ARCHITECTURE DATA SET 4

echo 'NUM NODES: 2';
./run.sh 1  ./main.py -df 4 -a 2 2 -e ARCHITECTURE -ss 4 -sh;
./run.sh 19 ./main.py -df 4 -a 2 2 -e ARCHITECTURE -ss 4;
echo 'NUM NODES: 4';
./run.sh 20 ./main.py -df 4 -a 4 4 -e  ARCHITECTURE -ss 4;
echo 'NUM NODES: 8';
./run.sh 20 ./main.py -df 4 -a 8 8 -e  ARCHITECTURE -ss 4;
echo 'NUM NODES: 16';
./run.sh 20 ./main.py -df 4 -a 16 16 -e  ARCHITECTURE -ss 4;
echo 'NUM NODES: 32';
./run.sh 20 ./main.py -df 4 -a 32 32 -e  ARCHITECTURE -ss 4;
./experiments.py -f num_nodes_data_4 -a;

# signal vs symbol data set 3

# echo 'SIGNAL';
# ./run.sh 1  ./main.py -s signal -df 3 -a 4 4 -e TYPE -ss 4 -sh;
# ./run.sh 19 ./main.py -s signal -df 3 -a 4 4 -e TYPE -ss 4;
# echo 'SYMBOL';
# ./run.sh 20 ./main.py -df 3 -a 4 4 -e  TYPE -ss 4;
# ./experiments.py -f sig_v_sym_data_3 -a;
