import sys;
import math;
reload(sys)
sys.setdefaultencoding="utf-8"
if (len(sys.argv)<3):
    print('Insufficient Parameters!');
    exit();
hownet_filename = sys.argv[1];
target_filename = sys.argv[2];
with open(hownet_filename,'r') as hownet:
    sememe_all = {};
    data = hownet.readlines();
    words = data[0::2];
    sememes = data[1::2];
    word2sememe = {};
    sem2semMatrix = {};
    seme_occur = 0;
    for word,sememe in list(zip(words,sememes)):
        sememe = sememe.strip().split();
        seme_occur += len(sememe);
        word2sememe[word.strip()] = sememe;
        for item in sememe:
            if (item in sememe_all):
                sememe_all[item] += 1;
            else:
                sememe_all[item] = 1;
        for item in sememe:
            for item2 in sememe:
                if (item == item2):
                    sem2semMatrix[(item,item2)] = 0;
                else:
                    if ((item,item2) in sem2semMatrix) :
                        sem2semMatrix[(item,item2)]+=1;
                        sem2semMatrix[(item2,item)]+=1;
                    else:
                        sem2semMatrix[(item,item2)] = 1;
                        sem2semMatrix[(item2,item)] = 1;
    seall_key = sememe_all.keys();
    with open(target_filename,'w') as target:
        with open('sememe_all','w') as seall:
            seall.write(str(len(seall_key))+'\n'+" ".join(list(seall_key)));
            for item in seall_key:
                for item2 in seall_key:
                    if (item == item2):
                        if (sememe_all[item] == 0):
                            target.write("0 ");
                        else:
                            target.write(str(math.log(1+float(sememe_all[item])/seme_occur))+" ");
                    else:
                        if ((item,item2) in sem2semMatrix):
                            target.write(str(math.log(1+float(sem2semMatrix[(item,item2)])/float(sememe_all[item])/float(sememe_all[item2])))+" ");
                        else: 
                            target.write("0 ");
                target.write('\n');
