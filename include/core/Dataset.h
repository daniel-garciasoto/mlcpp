//
// Created by danie on 06/11/2025.
//

#ifndef MLCPP_DATASET_H
#define MLCPP_DATASET_H
#include <optional>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

using namespace std;

namespace mlcpp {
    class Dataset {
    public:
        Dataset() = default;

        Dataset(vector<vector<double> > features, vector<int> labels);

        static optional<Dataset> from_csv(const string &filepath,bool has_header = true, int label_column = -1);
        pair<Dataset, Dataset> train_test_split(double test_ratio = 0.1, int seed = 41) const;

        void normalize();
        void standardize();

        const vector<vector<double> > &get_features() const { return this->features_; }
        const vector<int> &get_labels() const { return this->labels_; }
        size_t size() const { return this->features_.size(); }
        size_t num_features() const { return this->features_.empty() ? 0 : this->features_[0].size(); }

    private:
        vector<vector<double> > features_;
        vector<int> labels_;
    };
}


#endif //MLCPP_DATASET_H
