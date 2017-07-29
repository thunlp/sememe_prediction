#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdlib>
using std::string;
using std::vector;

std::unordered_map<string, vector<string>> hownets;

vector<string> labeled_words;
vector<string> unlabeled_words;

vector<vector<double>> word_embeddings;
vector<vector<double>> sememe_embeddings;
vector<vector<double>> word_sememe_matrix;
vector<vector<double>> gradsq;
vector<string> sememes_vec;
std::unordered_map<string, vector<double>> embeddings;
//std::unordered_map<string, vector<double>> sememe_embeddings;
//std::unordered_map<string, vector<double>> gradsq;
std::unordered_map<string, vector<std::pair<string, double>>> nearest_words;
std::unordered_set<string> sememes;
std::unordered_map<string, std::unordered_map<string, double>> scores;
double c = 0.8;
int DIM_SIZE = 0;
double lambda = 0.5;
int Nearest_K = 100;
double LEARNING_RATE = 0.01;
double ZERO_DECOMP_PROB = 0.005;
int MAX_ITER = 20;
string output_file("result.txt");
template<typename T>
T MyMin(T a, T b){
    return a < b ? a : b;
}

// load hownet database with words and their sememes and each word and its sememes are in the same line
bool ReadHowNet(const char* filename){
    std::ifstream fileHandler(filename);
    if(!fileHandler)
        return false;
    string line;
    while(getline(fileHandler, line)){
        std::stringstream ss(line);
        string word, sememe;
        ss >> word;
	    getline(fileHandler, line);
        std::stringstream ss2(line);
	    while(ss2 >> sememe){
            hownets[word].push_back(sememe);
	        sememes.insert(sememe);
	    }
    }
    for(auto & sememe : sememes)
	sememes_vec.push_back(sememe);
    fileHandler.close();
    return true;
}

// load word embeddings from word2vec and split them into labeled words and unlabeled words
bool ReadEmbeddings(const char* filename){
    std::ifstream fileHandler(filename);
    std::cout << "Begin read embeddings\n";
    if(!fileHandler)
        return false;
    string line;
    int word_size, dim;
    fileHandler >> word_size >> dim;
    DIM_SIZE = dim;
    for(size_t index = 0; index < word_size; index += 1){
        string word;
        vector<double> embedding(dim, 0.0);
        fileHandler >> word;
        double value = 0.0;
        for(size_t dim_index = 0; dim_index < dim; dim_index += 1){
            fileHandler >> embedding[dim_index];
            value += pow(embedding[dim_index], 2);
        }
        for(size_t dim_index = 0; dim_index < dim; dim_index += 1)
            embedding[dim_index] /= sqrt(value);
        embeddings[word] = embedding;
        if(hownets.find(word) != hownets.end()){
            labeled_words.push_back(word);
	        embedding.push_back(((float)rand()/RAND_MAX - 0.5)/ DIM_SIZE);
	        word_embeddings.push_back(embedding);
	    }
        else
            unlabeled_words.push_back(word);
    }
    for(auto word : labeled_words){
        vector<double> localsememes(sememes.size(), 0.0);
        for(auto sememe : hownets[word]){
            for(int index = 0; index < sememes_vec.size(); index ++){
                if(sememe == sememes_vec[index]){
                    localsememes[index] = 1.0;
                    break;
                }
            }
	    }
	    word_sememe_matrix.push_back(localsememes);
    }
    std::cout << "finish read embeddings\n";
    return true;
}



void CalculateNearestWords(){
    std::cout << "Begin calcuate nearest words\n";
    int count = 0;
    for(auto & word : unlabeled_words){
        count += 1;
        auto & vec1 = embeddings[word];
        vector<std::pair<float, string>> distances;
        for(auto & neighbor : labeled_words){
            double sum = 0.0;
            auto & vec2 = embeddings[neighbor];
            for(size_t dim_index = 0; dim_index < DIM_SIZE; dim_index ++){
                sum += vec1[dim_index] * vec2[dim_index];
            }
            distances.push_back(make_pair( 0 - sum, neighbor));
        }
        std::sort(distances.begin(), distances.end());
        auto & vec3 = nearest_words[word];
        for(size_t index = 0; index < Nearest_K; index += 1){
            vec3.push_back(make_pair(distances[index].second, 0 - distances[index].first));
        }
    }
}
void CalculateScoreFromEmbeddings(){
    for(auto & word : unlabeled_words){
        auto& neighbors = nearest_words[word];
        for(size_t rank = 0; rank < Nearest_K; rank += 1){
            string neighbor = neighbors[rank].first;
            double dist = neighbors[rank].second;
            for(auto & sememe : hownets[neighbor]){
                scores[word][sememe] += pow(dist, 2) * pow(c, rank);
            }
        }
    }
}

void CalculateScoreFromFactorization(){
    for(auto & word : unlabeled_words){
        for(int index = 0; index < sememes.size(); index ++){
            auto& sememe = sememes_vec[index];
            auto& vec1 = embeddings[word];
            auto& vec2 = sememe_embeddings[index];
            double sum = 0;
            for(size_t dim = 0; dim < DIM_SIZE; dim++)
                sum += vec1[dim] * vec2[dim];
            scores[word][sememe] += sum * lambda;
        }
    }
}
void CalculateSememeEmbeddings(){
    for(auto sememe : sememes){
        vector<double> vec(DIM_SIZE, 0.0);
        for(size_t dim_index = 0; dim_index < DIM_SIZE + 1; dim_index += 1){
            vec[dim_index] = (rand()/(double)RAND_MAX - 0.5) / DIM_SIZE;
        }
        sememe_embeddings.push_back(vec);
        gradsq.push_back(vector<double>(DIM_SIZE + 1, 1.0));
    }
    std::cout << "Begin training sememe embeddings\n";
    for(size_t iter = 0; iter < MAX_ITER; iter += 1){
        double count = 0;
        double total_loss = 0.0;
        for(size_t x = 0; x < labeled_words.size(); x++){
            for(size_t y = 0; y < sememes_vec.size(); y++){
                if(word_sememe_matrix[x][y] == 1.0 || (float)rand() / RAND_MAX > (1 - ZERO_DECOMP_PROB)){
                    count += 1;
                    double sum = 0;
                    for(size_t index = 0; index < DIM_SIZE; index += 1)
                        sum += word_embeddings[x][index] * sememe_embeddings[y][index];
                    sum += word_embeddings[x][DIM_SIZE] + sememe_embeddings[y][DIM_SIZE];
                    double loss = sum - word_sememe_matrix[x][y];
                    total_loss += pow(loss, 2);
                    for(size_t index = 0; index < DIM_SIZE; index += 1){
                        sememe_embeddings[y][index] -= loss * LEARNING_RATE * word_embeddings[x][index] / sqrt(gradsq[y][index]);
                        gradsq[y][index] += pow(loss * sememe_embeddings[y][index], 2);
                    }
                    word_embeddings[x][DIM_SIZE] -= loss / sqrt(gradsq[y][DIM_SIZE]);
                    sememe_embeddings[y][DIM_SIZE] -= loss / sqrt(gradsq[y][DIM_SIZE]);
                    gradsq[y][DIM_SIZE] += loss * loss;
                }
            }
        }
    }
    for(int i = 0; i < sememe_embeddings.size(); i++){
        auto& vec = sememe_embeddings[i];
        double sum = 0;
        for(auto value : vec)
	        sum += value * value;
        sum = sqrt(sum);
        for(auto& value: vec)
	        value /= sum;
    }
}

void output(char* output){
    std::ofstream fout(output);
    for(auto & word : unlabeled_words){
        vector<std::pair<double, string>> sememes;
        for(auto p : scores[word]){
            sememes.push_back(make_pair(0 - p.second, p.first));
        }
        std::sort(sememes.begin(), sememes.end());
        fout << word << std::endl;
        for(size_t rank = 0; rank < MyMin(10ul, sememes.size()); rank += 1){
            fout << "      " << sememes[rank].second << " " << 0 - sememes[rank].first << "\n";
        }
	    fout << "\n\n";
    }
}

int main(int argc, char** argv){
    if(argc < 4){
	std::cout << "No enough arguments\n";
	return 1;
    }
    // argv[1] is the file storing words and their sememes with the following format
    // word1
    // sememe11 sememe12 ...
    // word2
    // sememe21 sememe22 ...
    // ...
    ReadHowNet(argv[1]);
    // argv[2] stores the word embedding files with the following format
    // word_size dim_size
    // word1 embedding11 embedding12 ...
    // word2 embedding21 embedding22 ...
    // ...
    ReadEmbeddings(argv[2]);
    CalculateNearestWords();
    CalculateScoreFromEmbeddings();
    CalculateSememeEmbeddings();
    CalculateScoreFromFactorization();
    // argv[3] stores the result, namely sememe predicitons for unlabeled words 
    output(argv[3]);
    return 0;
}