import sys
import numpy
import pickle
import math

def Matrix_Factorization(embedding_matrix,annotation_table,sememe_size,epoch = 30,alpha = 0.01):
    shape = embedding_matrix.shape
    word_size = shape[0]
    vector_size = shape[1]
    sememe_embedding_matrix = numpy.random.rand(sememe_size,vector_size)-0.5
    gradsq_sememe_embedding_matrix = numpy.ones((sememe_size,vector_size))
    weight_matrix = numpy.random.rand(word_size,sememe_size)
    gradsq_weight_matrix = numpy.ones((word_size,sememe_size))
    for step in range(epoch):
        cost = 0;
        for num in range(word_size):
            for dim in range(vector_size):
                weight = numpy.zeros((1,sememe_size))
                for item in annotation_table[num]:
                    weight[0][item] = weight_matrix[num][item]
                loss = 0
                for i in range(sememe_size):
                    loss += weight[0][i]*sememe_embedding_matrix[i][dim]
                loss = loss - embedding_matrix[num][dim]
                cost += loss * loss
                loss = loss * 2
                for item in annotation_table[num]:
                    weight_matrix[num][item] -= alpha*loss*sememe_embedding_matrix[item][dim]/math.sqrt(gradsq_weight_matrix[num][item])
                    gradsq_weight_matrix[num][item] += (loss * sememe_embedding_matrix[item][dim]) ** 2
                    sememe_embedding_matrix[item][dim] -= alpha*loss*weight[0][item]/math.sqrt(gradsq_sememe_embedding_matrix[item][dim])
                    gradsq_sememe_embedding_matrix[item][dim] += (loss*weight[0][item]) ** 2

        print("process:%f,error:%f\n" % (step/float(epoch),cost/float(word_size*vector_size)))    
    return sememe_embedding_matrix;
    
    
sememe_all_filename = sys.argv[1]
Hownet_filename = sys.argv[2]
embedding_filename = sys.argv[3]

sememes = []
sememe_size = 0
with open(sememe_all_filename,'r',encoding='utf-8') as sememe_all_file:
    sememe_size = int(sememe_all_file.readline())
    sememes = sememe_all_file.readline().strip().strip('[]').split(' ')
    sememes = [x.strip().strip('\'') for x in sememes]

print("Sememe files reading complete")
#sememe_all reading complete
Hownet_dict = {}
word_dict = {}
with open(Hownet_filename,'r',encoding='utf-8') as Hownet_file:
    buf = Hownet_file.readlines()
    num = 0
    while (num < len(buf)):
        word = buf[num].strip()
        Hownet_dict_buf = buf[num+1].strip().split()
        Hownet_dict[word] = [sememes.index(x) for x in Hownet_dict_buf]
        num += 2
#Hownet reading complete
print("Hownet reading complete")

word_size = 0
vector_size = 0
embedding_matrix = []
embedding_matrix_all = {} 
annotation_table = []
with open(embedding_filename,'r',encoding='utf-8') as embedding_file:
    buf = embedding_file.readlines()
    num = 0
    word_size,vector_size = [int(x) for x in buf[0].split()]
    num += 1
    while (num < len(buf)):
        word = buf[num].strip().split()[0]
        embedding_matrix_all[word] = numpy.array([float(x) for x in buf[num].strip().split()[1:]]).reshape(1,vector_size);
        word_dict[word] = num
        if (word not in Hownet_dict):
            num += 1;
            continue;
        embedding_matrix.extend([float(x) for x in buf[num].strip().split()[1:]])
        annotation_table.append(Hownet_dict[word])
        num += 1

embedding_matrix = numpy.array(embedding_matrix).reshape(int(len(embedding_matrix)/vector_size),vector_size)
#embedding reading complete
print("Embedding reading complete")

print("Training start")
#with open("result_SPASE","rb") as result:
    #Sememe_embedding_Matrix = pickle.load(result)
Sememe_embedding_Matrix = Matrix_Factorization(embedding_matrix,annotation_table,sememe_size)

print("Backuping...")
with open("model_SPASE","wb") as result:
    pickle.dump(Sememe_embedding_Matrix,result)
print("Training finish, start evaluating")

test_filename = sys.argv[4]
output_file = open("output_SPASE",'w',encoding='utf-8')
print(Sememe_embedding_Matrix.shape)
with open(test_filename,'r',encoding='utf-8') as test_file:
    buf = test_file.readlines()
    num = 0 
    while (num < len(buf)):
        word = buf[num].strip()
        vec = embedding_matrix_all[word]
        result = []
        num_sememe = 0
        while (num_sememe < sememe_size):
            sememe = Sememe_embedding_Matrix[num_sememe]
            tmp = numpy.dot(vec,sememe.T)
            if (tmp<0):
                num_sememe+=1
                continue
            len1 = vec.dot(vec.T)
            len2 = sememe.dot(sememe.T)
            tmp = tmp/numpy.sqrt(len1*len2)
            result.append((sememes[num_sememe],tmp))
            num_sememe+=1
        result.sort(key=lambda x:x[1],reverse=True)
        output_file.write(word+'\n'+" ".join([x[0] for x in result])+'\n')
        num+=1
output_file.close()
