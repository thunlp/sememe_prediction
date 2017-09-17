#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
using namespace std;
int main(int argc,char* argv[]){
    ifstream fin1(argv[1]);
    ifstream fin2(argv[2]);
    map<string,vector<string> > result;
    vector<string> test;
    string word1,word2;
    float all=0,right=0, value; string line;
    while(fin1>>word1){
        vector<string> sememes;
        getline(fin1,line); getline(fin1,line);
        stringstream ss(line);
        while(ss>>line) sememes.push_back(line);ss>>line;
        result[word1]=sememes;
    }
    int k=0;
    while(fin2>>word1){
        getline(fin2,line);
        getline(fin2,line);
        vector<string> sememes=result[word1];
        if(sememes.size()==0) continue;
        stringstream ss(line);
        int j=0; int N=1;
        vector<string> sememeV;
        while(ss>>word2) sememeV.push_back(word2);
        float right2=0; int k=0;
        for(j=0;j<sememes.size();j++){
            int i;
            for(i=0;i<sememeV.size();i++)
                if(sememeV[i]==sememes[j])
                break;
            if(i!=sememeV.size()){
                k++;
                right+=float(k)/float(j+1);
                right2+=float(k)/float(j+1);
            }
        }
        all+=sememeV.size();
    }
    cout << "MAP: "<<float(right)/float(all)<<endl;
    return 0;
}