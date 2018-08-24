from __future__ import division
import sys
import pickle
import copy
import math

#default_encoding="utf-8"
#if(default_encoding!=sys.getdefaultencoding()):
    #reload(sys)
    #sys.setdefaultencoding(default_encoding)

if (len(sys.argv)<5):
    print ("Parameter insufficient!")
    exit()
embedding_filename = sys.argv[1]
hownet_filename = sys.argv[2]
test_filename = sys.argv[3]
output_filename = sys.argv[4]
embedding_file = None
hownet_file = None
output_file = None
def ScorerForSememe(target):
    vec = embedding_vec[target]
    res = copy.deepcopy(sememe_all)
    #for word in res:
        #res[word] = 0
    nearestwords = []
    for word in embedding_vec:
        if (word==target):
            continue
        if (word not in word2sememe):
            continue
        if (word in test_data):
            continue
        wordvec = embedding_vec[word]
        dotsum = sum([x*y for x,y in zip(wordvec,vec)])
        cosine = dotsum
        nearestwords.append((word,cosine))
    nearestwords.sort(key=lambda x:x[1],reverse=True)
    nearestwords = nearestwords[0:para_nearest_k]
    rank = 1
    for word,cosine in nearestwords:
        sememes = word2sememe[word]
        for sememe in sememes:
            res[sememe]+=cosine*pow(para_c,rank)
        rank+=1
    reslist = []
    for sememe in res:
        reslist.append((sememe,res[sememe]))
    reslist.sort(key=lambda x:x[1],reverse=True)
    final = []
    for sememe,score in reslist:
        final.append(sememe)
    return final,reslist 

try:
    embedding_file = open(embedding_filename,"r",encoding='utf-8')
    hownet_file = open(hownet_filename,"r",encoding='utf-8')
    output_file = open(output_filename,"w",encoding='utf-8')
    test_file = open(test_filename,"r",encoding='utf-8')
except Exception as e:
    print(e)
else:
    para_c = 0.8
    para_nearest_k = 100

    word_size = 0
    dim_size = 0
    print('Loading Embedding Files...')

    line = embedding_file.readline()
    arr = line.strip().split()

    word_size = int(arr[0])
    dim_size = int(arr[1])
    embedding_vec = {}
    for line in embedding_file:
        arr = line.strip().split()
        float_arr = []
        for i in range(1,dim_size+1):
            float_arr.append(float(arr[i]))
        regular = math.sqrt(sum([x*x for x in float_arr]))
        word = arr[0].strip()
        embedding_vec[word] = []
        for i in range(1,dim_size+1):
            embedding_vec[word].append(float(arr[i])/regular)
    print('Embedding File Successfully Loaded.')

    word2sememe = {}
    sememe_all = {}
    print('Loading Hownet File...')
    while True:
        word = hownet_file.readline().strip()
        sememes = hownet_file.readline().strip().split()
        if (word or sememes):
            word2sememe[word] = [] 
            length = len(sememes)
            for i in range(0,length):
                word2sememe[word].append(sememes[i])
                #print(word2sememe[word],len(word2sememe[word]))
                sememe_all[sememes[i]] = 0
        else: break
    print('Hownet File Successfully Loaded.')
    test_list = []
    test_data = test_file.readlines()
    checkBuffer = []
    print('Loading test data...')
    for word in test_data:
        if (word.strip() in embedding_vec):
            checkBuffer.append(word.strip())
    test_data=checkBuffer
    print('Test Data Successfully Loaded.')
    print("Initialization Complete.")
    model_file = open('model_SPWE','wb')
    for line in test_data:
        print("Process:%f" %(float(len(test_list)) / len(test_data)))
        test_list.append(line.strip())
        result=[]
        result,reslist = ScorerForSememe(line.strip())
        output_file.write(line.strip()+"\n")
        output_file.write(' '.join(result)+"\n")
        pickle.dump(reslist,model_file)
    output_file.close()
    test_file.close()
    embedding_file.close()
    hownet_file.close()
    model_file.close()
