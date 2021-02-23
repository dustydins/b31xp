#!/usr/bin/env bash

rm results/current_test.csv;

./run.sh 1 ./main.py -ss 1 -sh True;
./run.sh 19 ./main.py -ss 1;
./run.sh 20 ./main.py -ss 2;
./run.sh 20 ./main.py -ss 4;
./run.sh 20 ./main.py -ss 6;
./run.sh 20 ./main.py -ss 8;
./run.sh 20 ./main.py -ss 10;
