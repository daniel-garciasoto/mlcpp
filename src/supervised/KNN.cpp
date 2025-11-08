//
// Created by danie on 06/11/2025.
//

#include "../../include/supervised/KNN.h"

#include <map>
using namespace std;
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
        this->X_train_ = dataset.get_features();
        this->y_train_ = dataset.get_labels();
    }

    // Predict label for a single sample
    // Returns the predicted label
    int KNN::predict(const vector<double> &sample) const {
        vector<size_t> k_nearest_idx = find_k_nearest(sample);
        return majority_vote(k_nearest_idx);
    }

    // Predict labels for multiple samples
    // Returns vector of predicted labels
    vector<int> KNN::predict(const vector<vector<double> > &samples) const {
        vector<int> predicted_labels;
        predicted_labels.reserve(samples.size());
        for (size_t i = 0; i < samples.size(); i++) {
            vector<size_t> k_nearest_idx = find_k_nearest(samples[i]);
            predicted_labels.push_back(majority_vote(k_nearest_idx));
        }
        return predicted_labels;
    }

    // Calculate accuracy on a test dataset
    // Returns accuracy as a value between 0.0 and 1.0
    double KNN::score(const Dataset &test_dataset) const {
        // 1. Get the features and the labels from the test dataset
        const vector<vector<double>> X_test = test_dataset.get_features();
        const vector<int> y_test = test_dataset.get_labels();

        // 2. Validate that there is data
        if (y_test.empty() || X_test.empty()) {
            return 0.0;
        }

        // 3. Predict the test samples
        vector<int> y_pred = predict(X_test);

        // 4. Count correct predictions
        int correct = 0;
        for (size_t i = 0; i < y_test.size(); i++) {
            if (this->y_train_[i] == y_test[i]) {
                correct++;
            }
        }
        // 5. Calculate the accuracy
        return static_cast<double>(correct) / y_test.size();
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
            int label = this->y_train_[neighbor_indices[i]];
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
