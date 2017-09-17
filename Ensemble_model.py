import pickle;
import sys;
import math;
import numpy as np;
if (len(sys.argv)<4):
    print("Parameters insufficients");
hownet_filename = sys.argv[4];
model1_filename = sys.argv[1];
model2_filename = sys.argv[2];
ratio = float(sys.argv[3]);
with open(model1_filename,'rb') as model1_file:
    with open(model2_filename,'rb') as model2_file:
        model1 = [];
        model2 = [];
        print('Loading Models...')
        while True:
            try:
                model1.append(pickle.load(model1_file));
                model2.append(pickle.load(model2_file));
            except EOFError:
                break;
        print('Loading Models Complete, have read %d results from model1, %d results from model2' % (len(model1),len(model2)))
        assert(len(model1) == len(model2))
        index = 0;
        length = len(model1);
        test_words = [];
        print('Loading test files')
        with open(hownet_filename,'r') as test:
            for line in test:
                test_words.append(line.strip());
        print('Loading Complete,training beginning.')
        with open('output_Ensemble','w') as output:
            while (index < length):
                predict0 = dict(model1[index]);
                predict1 = dict(model2[index]);
                predict = [];
                for key in predict0:
                    predict.append((key,abs(ratio/(1+ratio)*(predict0[key])+1/(1+ratio)*predict1[key])));
                predict.sort(key=lambda x:x[1],reverse=True);
                result = [x[0] for x in predict];
                output.write(test_words[index]+'\n'+" ".join(result)+'\n');
                index += 1;
        print('Training complete.')        
            
            
