//
// Created by danie on 06/11/2025.
//

#include "../../include/supervised/KNN.h"

#include <map>

namespace mlcpp {
    // Constructor
    // k: number of neighbors to consider
    // distance: distance metric function
    KNN::KNN(int k, mlcpp::DistanceMetric distance) {
        this->k_ = k;
        this->distance_ = distance;
    }

    // Fit the model with training data
    // This is a "lazy" algorithm - just stores the training data
    void KNN::fit(Dataset &dataset) {
    }

    // Predict label for a single sample
    // Returns the predicted label
    int KNN::predict(const vector<double> &sample) const {
        return 0;
    }

    // Predict labels for multiple samples
    // Returns vector of predicted labels
    vector<int> KNN::predict(const vector<vector<double> > &samples) const {
        return {};
    }

    // Calculate accuracy on a test dataset
    // Returns accuracy as a value between 0.0 and 1.0
    double KNN::score(const Dataset &test_dataset) const {
        return 0.0;
    }

    // Find indices of k nearest neighbors for a given sample
    vector<size_t> KNN::find_k_nearest(const vector<double> &sample) const {
        //Calculate the distances and their index and save them in vector of pairs
        vector<pair<double, size_t> > distances;
        for (size_t i = 0; i < this->X_train_.size(); i++) {
            distances.push_back({this->distance_(sample, this->X_train_[i]), i});
        }

        //Sort it by the distances
        partial_sort(distances.begin(), distances.begin() + this->k_, distances.end());

        //Add the k_nearest to the vector
        vector<size_t> k_nearest;
        for (size_t i = 0; i < this->k_ - 1; i++) {
            k_nearest.push_back(distances[i].second);
        }
        return k_nearest;
    }

    // Get majority vote from neighbor labels
    int KNN::majority_vote(const vector<size_t> &neighbor_indices) const {

        map<size_t,size_t> label_counts;
        for (size_t i = 0; i < neighbor_indices.size(); i++) {
            int label = this->Y_train_[neighbor_indices[i]];
            label_counts[label]++;
        }

        // Find the label with maximum count
        int max = 0;
        int best_label = 0;
        for (const auto& [label, count] : label_counts) {
            if (count > max) {
                max = count;
                best_label = label;
            }
        }
        return best_label;
    }
}
