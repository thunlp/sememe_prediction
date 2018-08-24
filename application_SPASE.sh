cp embedding_200.txt train_data
cp hownet.txt train_hownet
python SPASE.py sememe_all train_hownet train_data hownet.txt_test
