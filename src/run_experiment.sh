#!/usr/bin/env bash

rm results/current_test.csv;

echo 'SUBSEQUENCE: 10';
./run.sh 1  ./main.py -a 32 32 -e SUBSEQUENCE_SIZE -ss 10 -sh;
./run.sh 19 ./main.py -a 32 32 -e SUBSEQUENCE_SIZE -ss 10;
echo 'SUBSEQUENCE: 15';
./run.sh 20 ./main.py -a 32 32 -e  SUBSEQUENCE_SIZE -ss 15;
echo 'SUBSEQUENCE: 20';
./run.sh 20 ./main.py -a 32 32 -e  SUBSEQUENCE_SIZE -ss 20;
echo 'SUBSEQUENCE: 25';
./run.sh 20 ./main.py -a 32 32 -e  SUBSEQUENCE_SIZE -ss 25;

./experiments.py -f subsequence_ext_linear_mlp -a;
