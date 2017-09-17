#Example for SPWE && SPSE, other combinations follow the same pattern
#Make sure that you have run SPSE.sh && SPWE.sh
python Ensemble_model.py model_SPWE model_SPSE 2.1 hownet.txt_test
python scorer.py output_Ensemble hownet.txt_answer
