cp embedding_200.txt train_data
cp hownet.txt train_hownet
python SPWE.py train_data train_hownet hownet.txt_test output_SPWE
