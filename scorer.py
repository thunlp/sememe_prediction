from __future__ import division
from __future__ import print_function
import sys;
reload(sys)
sys.setdefaultencoding='utf-8'
if (len(sys.argv)<3):
    print("not enough parameters!");
    exit();
test_filename = sys.argv[1];
answer_filename = sys.argv[2];
scores = [];
with open(test_filename,'r',encoding='utf-8') as test:
    with open(answer_filename,'r',encoding='utf-8') as answer:
        while (True):
            test_word = test.readline().strip();
            answer_word = answer.readline().strip();
            if (len(test_word)==0 or len(answer_word)==0):
                break;
            while (test_word != answer_word):  #some word not exist in embeddings
                answer.readline();
                answer_word = answer.readline().strip();
            #print(test_word);
            #print(answer_word);
            test_sememes = test.readline().strip().strip('[]').split(' ');
            answer_sememes = answer.readline().strip().strip('[]').split(' ');
            point = 0;
            length = len(test_sememes);
            #for i in range(0,length):
                ##print(test_sememes[i]);
                #test_sememes[i] = test_sememes[i].strip().strip('\'');
                ##print(i);
            if (len(answer_sememes)==0): 
                continue;
            #print(test_sememes);
            #print(answer_sememes);
            index = 1
            correct = 0
            for item in (test_sememes):
                #try:
                if (item in answer_sememes):
                   correct += 1
    	   #if (index == 1):
    	       #print(item)
    	       #print(answer_sememes);
    	       #print(test_sememes);
                   #print(rank);
                   point += float(correct) / (index);
                   index+=1;
                #except:
            point /= len(answer_sememes);
            #print(point);
            scores.append(point);
print("result:%f" % (sum(scores)/len(scores),));
