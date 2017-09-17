#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <cmath>
#include <sstream>
#include <unordered_map>
using namespace std;

#define LEARNING_RATE 0.01
int sememe_size;
int word_size;
int embedding_dim;

vector<string> sememes;
vector<vector<int>> word_sememe_matrix;
vector<vector<int>> train_word_sememe_matrix;
vector<string> words;
vector<vector<float>> word_embeddings;
float *W;
float *gradsq;
unordered_map<string, vector<float>> total_word_embeddings;
bool init(char* word_sememe_file, char* sememe_file, char* word_embeddings_file){
    cout << "Loading all sememes\n";
    ifstream fin(sememe_file);
    fin >> sememe_size;
    string sememe;
    while(fin >> sememe)
        sememes.push_back(sememe);
    fin.close();

    cout << "Loading word sememes\n";
    fin.open(word_sememe_file);
    string line;
    while(getline(fin, line)){
        words.push_back(line);
        getline(fin, line);
        istringstream ss(line);
        vector<int> word_sememe;
        while(ss >> sememe){
            int index = 0;
            for(int i = 0; i < sememe_size; i++){
                if(sememes[i] == sememe){
                    index = i;
                    break;
                }
            }
            word_sememe.push_back(index);
        }
        word_sememe_matrix.push_back(word_sememe);
    }
    word_size = word_sememe_matrix.size();
    fin.close();

    cout << "Loading word embeddings.\n";
    fin.open(word_embeddings_file);
    int _word_size;
    fin >> _word_size >> embedding_dim;
    int count = 0;
    string word;
    while(fin >> word){
        vector<float> vec(embedding_dim, 0.0);
        double sum = 0.0;
        for(int j = 0; j < embedding_dim; j++){
            fin >> vec[j];
            sum += vec[j] * vec[j];
        }
        sum = sqrt(sum);
        for(int j = 0; j < embedding_dim; j++)
            vec[j] /= sum;
        int index = -1;
        for(int j = 0; j < words.size(); j++){
            if(words[j] == word){
                index = j;
                break;
            }
        }
        if(index != -1){
            word_embeddings.push_back(vec);
            train_word_sememe_matrix.push_back(word_sememe_matrix[index]);
        }
        total_word_embeddings[word] = vec;
    }
    word_size = word_embeddings.size();
    W = new  float[sememe_size * (embedding_dim + 1)];
    gradsq = new  float[sememe_size * (embedding_dim + 1)];

    if(!W || !gradsq){
        cerr<<"Error to init the matrix\n";
        return false;
    }
    // initialize the sememe matrix.

    for(int i = 0; i < sememe_size; i++)
        for(int j = 0; j < embedding_dim + 1; j++)
            W[i * (embedding_dim + 1) + j] = (rand() / ( float)RAND_MAX - 0.5) / sememe_size;

    // for adagrad training
    for(int i = 0; i < sememe_size; i++)
        for(int j = 0; j < embedding_dim + 1; j++)
            gradsq[i * (embedding_dim + 1) + j] = 1.0;
    return true;
}

void train(int epoch_num){
    vector<int> word_indexes;
    for(int i = 0; i < word_size; i++){
        word_indexes.push_back(i);
    }
    for(int epoch = 0; epoch < epoch_num; epoch ++){
        cout << "Training at epoch " << epoch << std::endl;
        for(int i = 0; i < word_size; i++)
            std::swap(word_indexes[i], word_indexes[rand() % word_size]);
         float cost = 0.0;
        for(int word_id = 0; word_id < word_size; word_id ++){
            int index = word_indexes[word_id];
            for(int dim = 0 ; dim < embedding_dim; dim ++){
                 float diff = 0.0;
                for(auto & sememe_id : train_word_sememe_matrix[index]){
                    diff += W[sememe_id * (embedding_dim + 1) + dim];
                }
                diff -= word_embeddings[index][dim];
                cost += diff * diff;
                for(auto & sememe_id : train_word_sememe_matrix[index]){
                    auto place = sememe_id * (embedding_dim + 1) + dim;
                    W[place] -= LEARNING_RATE * diff / sqrt(gradsq[place]);
                    gradsq[place] += diff * diff;
                }
            }
        }
        cout << "cost : " << sqrt(cost / (word_size * embedding_dim)) << std::endl;
    }
}

void save(char* save_file, char* sememe_file){
    ofstream fout(save_file);
    ifstream fin(sememe_file);
    int _sememe_size;
    fin >> _sememe_size;
    string sememe;
    fout << _sememe_size << " " << embedding_dim << endl;
    int count = 0;
    while(fin >> sememe){
        fout << sememe << " ";
        for(int i = 0; i < embedding_dim; i++)
            fout << W[count * (embedding_dim + 1) + i] << " ";
        fout << endl;
        count++;
    }
    fin.close();
    fout.close();
}

void computeSimilarity(char* hownet_test_file, char* result_file){
    ifstream fin(hownet_test_file);
    string word;
    ofstream fout(result_file);
    for(int i = 0; i < sememe_size; i++){
        double sum = 0;
        for(int j = 0; j < embedding_dim; j++){
            sum += pow(W[i * (embedding_dim + 1) + j], 2);
        }
        sum = sqrt(sum);
        for(int j = 0; j < embedding_dim; j++)
            W[i * (embedding_dim + 1) + j] /= sum;
    }
    while(fin >> word){
        if(total_word_embeddings.find(word) == total_word_embeddings.end()){
            cout << "Can not find " << word << endl;
            continue;
        }
        auto & word_vec = total_word_embeddings[word];
        double sum = 0;
        for(int i = 0; i < embedding_dim; i++)
            sum += pow(word_vec[i], 2);
        sum = sqrt(sum);
        for(int i = 0; i < embedding_dim; i++)
            word_vec[i] /= sum;
        vector<pair<float, string>> vec;
        for(int i = 0; i < sememe_size; i++){
            sum = 0;
            for(int j = 0; j < embedding_dim; j++)
                sum += word_vec[j] * W[i * (embedding_dim + 1) + j];
            vec.push_back({ 0 - sum, sememes[i]});
        }
        sort(vec.begin(), vec.end());
        fout << word << endl;
        for(int i = 0; i < 20; i++)
            fout << "    " << vec[i].second << " " << 0 - vec[i].first << " ";
        fout << endl;
    }
    fout.close();
    fin.close();
}
int main(int argc, char**argv){
    if(init(argv[1], argv[2], argv[3])){
        train(atoi(argv[4]));
    }
    save(argv[5], argv[2]);
    computeSimilarity(argv[6], argv[7]);
    return 0;
}