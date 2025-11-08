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



namespace mlcpp {
    class Dataset {
    public:
        Dataset() = default;

        Dataset(std::vector<std::vector<double> > features, std::vector<int> labels);

        static std::optional<Dataset> from_csv(const std::string &filepath,bool has_header = true, int label_column = -1);
        std::pair<Dataset, Dataset> train_test_split(double test_ratio = 0.1, int seed = 41) const;

        void normalize();
        void standardize();

        const std::vector<std::vector<double> > &get_features() const { return this->features_; }
        const std::vector<int> &get_labels() const { return this->labels_; }
        size_t size() const { return this->features_.size(); }
        size_t num_features() const { return this->features_.empty() ? 0 : this->features_[0].size(); }

    private:
        std::vector<std::vector<double> > features_;
        std::vector<int> labels_;
    };
}


#endif //MLCPP_DATASET_H
