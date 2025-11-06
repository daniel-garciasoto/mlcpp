//
// Created by danie on 06/11/2025.
//

#ifndef MLCPP_KNN_H
#define MLCPP_KNN_H
#include <vector>
#include "../core/Dataset.h"
#include "../core/Distance.h"
using namespace std;

namespace mlcpp {
    class KNN {

    public:

        explicit KNN(int k = 3, DistanceMetric distance = euclidean_distance);
        void fit(Dataset& dataset);
        int predict(const vector<double>& sample) const;
        vector<int> predict(const vector<vector<double>>& samples) const;
        double score(const Dataset& test_dataset) const;
        int get_k() const {return k_;}
    private:
        int k_;
        DistanceMetric distance_;
        vector<vector<double>> X_train_;
        vector<int> Y_train_;

        vector<size_t> find_k_nearest(const vector<double>& sample) const;
        int majority_vote(const vector<size_t>& neighbor_indices) const;
    };
}

#endif //MLCPP_KNN_H