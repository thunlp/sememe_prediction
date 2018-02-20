import sys;
#reload(sys)
#sys.setdefaultencoding="utf-8"
if (len(sys.argv)<4):
    print('no enough parameter')
    exit();
hownet_filename = sys.argv[1];
embedding_filename = sys.argv[2];
target_filename = sys.argv[3];
with open(hownet_filename,'r',encoding='utf-8') as hownet:
    with open(embedding_filename,'r',encoding='utf-8') as embedding:
        with open(target_filename,'w',encoding='utf-8') as target:
            wordsBuf = embedding.readlines();
            dim_size = int(wordsBuf[0].strip().split()[1])
            dic = hownet.readlines();
            wordlen = len(wordsBuf) ;
            words = {};
            for i in range(1,wordlen):
               line = wordsBuf[i].strip().split();
               words[line[0].strip()] = i;
            index = 0;
            diclen = len(dic);
            Strings = [];
            while(index<diclen):
                now = dic[index].strip();
                if (now in words):
                    #target.write(wordsBuf[words[now]]);
                    Strings.append(wordsBuf[words[now]]);
                index+=2;
            target.write(str(len(Strings))+" "+str(dim_size)+"\n")
            for line in Strings:
                target.write(line);
            
