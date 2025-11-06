//
// Created by danie on 06/11/2025.
//

#ifndef MLCPP_DATASET_H
#define MLCPP_DATASET_H
#include <vector>

using namespace std;

class Dataset {
public:
    static void from_csv();
    static void train_test_split();
    static void normalize();
public:
    vector<vector<double>> features_;
    vector<int> labels_;

};


#endif //MLCPP_DATASET_H
