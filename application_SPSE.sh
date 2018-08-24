cp embedding_200.txt train_data
cp hownet.txt train_hownet
python SPSE_Training.py train_hownet train_data sememe_all SPSE_embedding
python SPSE_Prediction.py SPSE_embedding sememe_all train_data hownet.txt_test output_SPSE 
