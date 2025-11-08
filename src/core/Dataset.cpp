//
// Created by danie on 06/11/2025.
//

#include "../../include/core/Dataset.h"


namespace mlcpp {
    Dataset::Dataset(vector<vector<double> > features, vector<int> labels) {
        this->features_ = features;
        this->labels_ = labels;
    }

    static optional<Dataset> from_csv(const string &filepath, bool has_header = true, int label_column = -1) {
        // 1. Verify if it ends with .csv
        if (filepath.size() < 4 ||
            filepath.substr(filepath.size() - 4) != ".csv") {
            return {};
        }

        // 2. Open the file
        ifstream file(filepath);
        if (!file.is_open()) {
            return {};
        }

        // 3. Jump the line if it has a header
        string line;
        if (has_header) {
            getline(file, line);
        }

        // 4. Read data
        vector<vector<double> > features;
        vector<int> labels;

        while (getline(file, line)) {
            istringstream line_stream(line);
            string cell;
            vector<double> row; // Temporal row for features
            int col_index = 0;
            int num_cols = 0;

            // Count columns (only first row)
            if (features.empty()) {
                istringstream temp_stream(line);
                while (getline(temp_stream, cell, ',')) {
                    num_cols++;
                }
                line_stream.clear();
                line_stream.str(line); // Reset stream
            }

            // Determinate label's indexes
            int actual_label_col = label_column;
            if (label_column == -1) {
                actual_label_col = num_cols - 1; // Last column
            }

            int label = 0;
            bool label_found = false;

            // Read each cell
            while (getline(line_stream, cell, ',')) {
                try {
                    double value = stod(cell); // String to double
                    if (col_index == actual_label_col) {
                        // It's the label's column
                        label = static_cast<int>(value);
                        label_found = true;
                    } else {
                        // It's a feature
                        row.push_back(value);
                    }
                } catch (const invalid_argument &e) {
                    // Error
                    return {};
                }
                col_index++;
            }

            // Save the row and the label
            if (label_found) {
                features.push_back(row);
                labels.push_back(label);
            }
        }

        file.close();

        // 5. Verify that the data is read
        if (features.empty()) {
            return {};
        }

        // 6. Create and return the Dataset
        return Dataset(features, labels);
    }


    pair<Dataset, Dataset> Dataset::train_test_split(double test_ratio, int seed) const {
        // 1. Validate test_ratio
        if (test_ratio <= 0.0 || test_ratio >= 1.0) {
            throw invalid_argument("test_ratio must be between 0 and 1.");
        }

        // 2. Calculate the sizes
        size_t total_size = this->features_.size();
        size_t test_size = static_cast<size_t>(total_size * test_ratio);
        size_t train_size = total_size - test_size;

        // 3. Create indexes and shuffle them
        vector<size_t> indexes(total_size);
        for (size_t i = 0; i < total_size; i++) {
            indexes[i] = i;
        }

        mt19937 generator(seed);
        shuffle(indexes.begin(), indexes.end(), generator);

        //4. Separate the dataset in train and test
        vector<vector<double> > train_features;
        vector<int> train_labels;
        vector<vector<double> > test_features;
        vector<int> test_labels;

        for (size_t i = 0; i < train_size; i++) {
            size_t idx = indexes[i];
            train_features.push_back(this->features_[idx]);
            train_labels.push_back(this->labels_[idx]);
        }
        for (size_t i = train_size; i < total_size; i++) {
            size_t idx = indexes[i];
            test_features.push_back(this->features_[idx]);
            test_labels.push_back(this->labels_[idx]);
        }

        Dataset train_dataset = Dataset(train_features, train_labels);
        Dataset test_dataset = Dataset(test_features, test_labels);
        return {train_dataset, test_dataset};
    }


    void Dataset::normalize() {
        if (features_.empty()) {
            return; // There is no data
        }

        size_t num_samples = features_.size();
        size_t num_features = features_[0].size();

        // Normalize each feature (column) independently
        for (size_t col = 0; col < num_features; ++col) {
            // 1. Find min and max of each feature
            double min_val = features_[0][col];
            double max_val = features_[0][col];

            for (size_t row = 0; row < num_samples; ++row) {
                double value = features_[row][col];
                if (value < min_val) {
                    min_val = value;
                }
                if (value > max_val) {
                    max_val = value;
                }
            }

            // 2. Normalize: (x - min) / (max - min)
            double range = max_val - min_val;

            // Avoid dividing by zero
            if (range > 0.0) {
                for (size_t row = 0; row < num_samples; ++row) {
                    features_[row][col] = (features_[row][col] - min_val) / range;
                }
            }
            // If range == 0, let the values as they (they are all the same)
        }
    }

    void Dataset::standardize() {
        if (features_.empty()) {
            return; // There is data
        }

        size_t num_samples = features_.size();
        size_t num_features = features_[0].size();

        // Normalize each feature (column) independently
        for (size_t col = 0; col < num_features; ++col) {
            // 1. Calculate mean
            double total = 0;
            for (size_t row = 0; row < num_samples; ++row) {
                total += features_[row][col];
            }
            double mean = total / num_samples;

            // 2. Calculate standard deviation
            double variance_sum = 0.0;
            for (size_t row = 0; row < num_samples; ++row) {
                double diff = features_[row][col] - mean;
                variance_sum += diff * diff;
            }

            double std_dev = sqrt((variance_sum) / (num_samples - 1));

            // 3. Standardize: (x - mean) / std_dev
            // Avoid dividing by zero
            if (std_dev > 0.0) {
                for (size_t row = 0; row < num_samples; ++row) {
                    features_[row][col] = (features_[row][col] - mean) / std_dev;
                }
            }
            // If range == 0, let the values as they (they are all the same)
        }
    }
}
