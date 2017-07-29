#!/bin/sh
g++ sememe_predict.cpp -std=c++11 -O3 -o sememe_predict
./sememe_predict hownet.txt embedding_50.txt result.txt
