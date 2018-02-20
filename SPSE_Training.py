from __future__ import division
import numpy as np
import pickle
import random
import math
import sys
np.set_printoptions(threshold=np.nan)
#reload(sys)
#sys.setdefaultencoding("utf-8")
if (len(sys.argv)<5):
    exit(0)
hownet_filename = sys.argv[1]
embedding_filename = sys.argv[2]
sememe_all_filename = sys.argv[3]
target_filename = sys.argv[4]
para_lambda = 0
max_iter = 20

with open(hownet_filename,'r',encoding='utf-8') as hownet:
    with open(embedding_filename,'r',encoding='utf-8') as embedding_file:
        with open(sememe_all_filename,'r',encoding='utf-8') as sememe_all:
            sememes_buf = sememe_all.readlines() 
            sememes = sememes_buf[1].strip().strip('[]').split(' ')
            sememes = [sememe.strip().strip('\'') for sememe in sememes]
            sememe_size = len(sememes)
            hownet_words = []
            #read sememe complete
            word2sememe = {}
            while True:
                word = hownet.readline().strip()
                sememes_tmp = hownet.readline().strip().split()
                if (word or sememes_tmp):
                    word2sememe[word] = [] 
                    hownet_words.append(word)
                    length = len(sememes_tmp)
                    for i in range(0,length):
                        word2sememe[word].append(sememes_tmp[i])
                else: break 
            #read hownet complete
            print("hownet reading complete")
            line = embedding_file.readline()
            arr = line.strip().split()

            word_size = len(hownet_words)
            dim_size = int(arr[1])
            embedding_vec = {}
            W = []
            for line in embedding_file:
                arr = line.strip().split()
                float_arr = []
                now_word = arr[0].strip()
                if (now_word not in hownet_words):
                    continue
                for i in range(1,dim_size+1):
                    float_arr.append(float(arr[i]))
                regular = math.sqrt(sum([x*x for x in float_arr]))
                word = arr[0].strip()
                embedding_vec[word] = []
                for i in range(1,dim_size+1):
                    embedding_vec[word].append(float(arr[i])/regular)
                    W.append(float(arr[i])/regular)
            # sometimes, people use word_embeddings with fewer words than hownet
            W = np.array(W).reshape(-1,dim_size)
            word_size = len(embedding_vec)

            #read embedding complete
            print('Embedding reading complete')
            with open('PMI.txt','r',encoding='utf-8') as PMI:
                P = []
                for line in PMI:
                    arr = line.strip().split()
                    arr = [float(e) for e in arr]
                    P.extend(arr)
                P = np.array(P).reshape(sememe_size,sememe_size)
                M = np.zeros((word_size,sememe_size))
                se_index = 0
                word_index = 0
                for word in hownet_words:
                    if (word not in embedding_vec):
                        continue
                    try:
                        for sememe in word2sememe[word]:
                            se_index = sememes.index(sememe)
                            M[word_index][se_index] = 1
                        word_index += 1
                    except:
                        print(word)
                        sys.exit()
                print("PMI calculating complete")
            sememe_embedding = (np.random.randn(sememe_size*2,dim_size)-0.5) / dim_size
            bias_sememe = (np.random.randn(sememe_size,1)-0.5) / dim_size
            bias_word = (np.random.randn(word_size,1)-0.5) / dim_size
            try:
                print('Try to read from checkpoint')
                target=open(target_filename,'rb')
                sememe_embedding = pickle.load(target)
                bias_word = pickle.load(target)
                bias_sememe = pickle.load(target)
                print('checkpoint reading complete')
                target.close()
            except:
                print('checkpoint reading failed, initialize with random value')
                
            with open(target_filename,'wb') as target:
                sememe_embedding_dersum = np.ones((sememe_size*2,dim_size)) 
                bias_sememe_dersum = np.ones((sememe_size,1)) 
                bias_word_dersum = np.ones((word_size,1)) 
                print('Initailization complete')
                learning_rate = 0.01
                for i in range(1,max_iter):
                    print("Process:%f" %(i/max_iter))
                    loss = 0
                    count = 0
                    for j in range(0,word_size):
                        for i in range(0,sememe_size):
                            sem0 = sememe_embedding[2 * i]
                            sem1 = sememe_embedding[2 * i + 1]
                            der = np.zeros((1,dim_size))
                            if (M[j][i] == 0):
                                rand = random.randint(1,1000)
                                if (rand>5):
                                    continue
                            count += 1
                            w = W[j].reshape(1,dim_size)
                            delta = w.dot((sem0+sem1).transpose())+bias_sememe[i]+bias_word[j]-M[j][i]
                            loss += delta ** 2
                            der += delta * 2 * w
                            der = der.reshape(dim_size,)
                            sememe_embedding[2 * i] += -learning_rate * der / sememe_embedding_dersum[2 * i]
                            sememe_embedding[2 * i + 1] += -learning_rate * der / sememe_embedding_dersum[2 * i + 1]
                            sememe_embedding_dersum[2 * i] += der ** 2
                            sememe_embedding_dersum[2 * i + 1] += der ** 2
                            bias_word[j] += 2 * delta * learning_rate / bias_word_dersum[j]
                            bias_word_dersum[j] += 4 * delta ** 2
                            bias_sememe[i] += 2 * delta * learning_rate / bias_sememe_dersum[i]
                            bias_sememe_dersum[i] += 4 * delta ** 2
                    for j in range(0,sememe_size):
                        for i in range(0,sememe_size):
                            sem0 = sememe_embedding[2 * j]
                            sem1 = sememe_embedding[2 * i + 1]
                            der = np.zeros((1,dim_size))
                            der_out = np.zeros((1,dim_size))
                            if (P[j][i] == 0):
                                rand = random.randint(1,1000)
                                if (rand>5):
                                    continue
                            count += 1
                            w = W[j].reshape(1,dim_size)
                            delta = sem0.dot((sem1).transpose())-P[j][i]
                            loss += para_lambda * delta ** 2
                            
                            der += para_lambda * delta * 2 * sem0
                            der = der.reshape(dim_size,)
                            sememe_embedding[2 * i + 1] += -learning_rate * der / sememe_embedding_dersum[2 * i + 1]
                            sememe_embedding_dersum[2 * i + 1] += der ** 2
                            
                            der_out += para_lambda * delta * 2 * sem1
                            der_out = der_out.reshape(dim_size,)
                            sememe_embedding[2 * j] += -learning_rate * der_out / sememe_embedding_dersum[2 * j]
                            sememe_embedding_dersum[2 * j] += der_out ** 2
                            
                    print("loss:%f" %(loss / count,))
                pickle.dump(sememe_embedding,target)
                pickle.dump(bias_word,target)
                pickle.dump(bias_sememe,target)
                 
