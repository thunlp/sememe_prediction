#/bin/sh
g++ SPASE.cpp -std=c++11 -o3 -o SPASE
g++ score.cpp -std=c++11 -o3 -o score
./SPASE train_hownet sememe_all train_data 200 sememme_embedding.txt hownet.txt_test result.txt
./score result.txt hownet.txt_answer