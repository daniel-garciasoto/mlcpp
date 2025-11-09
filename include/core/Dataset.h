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
#include <map>

namespace mlcpp {
    /**
     * @brief Container class for machine learning datasets.
     *
     * Stores features (input data) and labels (output data) for supervised learning tasks.
     * Provides utilities for loading, splitting, and normalizing data.
     */
    class Dataset {
    public:
        /**
         * @brief Default constructor. Creates an empty dataset.
         */
        Dataset() = default;

        /**
         * @brief Constructs a dataset with given features and labels.
         *
         * @param features 2D vector where each row is a sample and each column is a feature
         * @param labels Vector of integer labels corresponding to each sample
         */
        Dataset(std::vector<std::vector<double>> features, std::vector<int> labels);

        /**
         * @brief Loads a dataset from a CSV file.
         *
         * Supports both numeric and text labels (text labels are automatically mapped to integers).
         * The CSV file should have features as columns and samples as rows.
         *
         * @param filepath Path to the CSV file
         * @param has_header Whether the first row contains column headers (default: true)
         * @param label_column Index of the label column, -1 for last column (default: -1)
         * @return Optional Dataset object. Returns empty optional if loading fails.
         *
         * @note Text labels are automatically converted to numeric IDs (0, 1, 2, ...)
         * @note All feature columns must contain numeric values
         *
         * Example usage:
         * @code
         * auto dataset = Dataset::from_csv("data/iris.csv", true, -1);
         * if (dataset) {
         *     cout << "Loaded " << dataset->size() << " samples" << endl;
         * }
         * @endcode
         */
        static std::optional<Dataset> from_csv(const std::string& filepath,
                                               bool has_header = true,
                                               int label_column = -1);

        /**
         * @brief Splits the dataset into training and testing sets.
         *
         * Randomly shuffles the data before splitting to ensure random distribution.
         * Uses a seed for reproducibility.
         *
         * @param test_ratio Proportion of data to use for testing (between 0.0 and 1.0, default: 0.1)
         * @param seed Random seed for reproducibility (default: 41)
         * @return Pair of datasets: {train_dataset, test_dataset}
         *
         * @throws std::invalid_argument If test_ratio is not between 0.0 and 1.0
         *
         * @note Time complexity: O(n) where n is the number of samples
         *
         * Example usage:
         * @code
         * auto [train, test] = dataset.train_test_split(0.2, 42);  // 80% train, 20% test
         * cout << "Train: " << train.size() << " samples" << endl;
         * cout << "Test: " << test.size() << " samples" << endl;
         * @endcode
         */
        std::pair<Dataset, Dataset> train_test_split(double test_ratio = 0.1,
                                                      int seed = 41) const;

        /**
         * @brief Normalizes all features to the range [0, 1] using min-max scaling.
         *
         * For each feature column:
         * normalized_value = (value - min) / (max - min)
         *
         * This modifies the dataset in-place. Features with constant values (range = 0)
         * are left unchanged.
         *
         * @note Use this when you need features in a bounded range [0, 1]
         * @note This is sensitive to outliers
         * @note Time complexity: O(n * d) where n = samples, d = features
         *
         * @see standardize() for an alternative normalization method
         *
         * Example usage:
         * @code
         * dataset.normalize();
         * // All features now in [0, 1] range
         * @endcode
         */
        void normalize();

        /**
         * @brief Standardizes all features to have mean=0 and standard deviation=1.
         *
         * For each feature column:
         * standardized_value = (value - mean) / std_dev
         *
         * This modifies the dataset in-place. Features with zero standard deviation
         * are left unchanged.
         *
         * @note Use this when features have different scales and you care about distribution
         * @note Less sensitive to outliers than normalize()
         * @note Time complexity: O(n * d) where n = samples, d = features
         *
         * @see normalize() for min-max scaling alternative
         *
         * Example usage:
         * @code
         * dataset.standardize();
         * // All features now have mean=0, std=1
         * @endcode
         */
        void standardize();

        /**
         * @brief Gets the feature matrix (read-only).
         *
         * @return Constant reference to the 2D feature vector
         */
        const std::vector<std::vector<double>>& get_features() const {
            return this->features_;
        }

        /**
         * @brief Gets the label vector (read-only).
         *
         * @return Constant reference to the label vector
         */
        const std::vector<int>& get_labels() const {
            return this->labels_;
        }

        /**
         * @brief Gets the number of samples in the dataset.
         *
         * @return Number of samples (rows)
         */
        size_t size() const {
            return this->features_.size();
        }

        /**
         * @brief Gets the number of features per sample.
         *
         * @return Number of features (columns), or 0 if dataset is empty
         */
        size_t num_features() const {
            return this->features_.empty() ? 0 : this->features_[0].size();
        }

    private:
        std::vector<std::vector<double>> features_;  ///< 2D array of features [samples][features]
        std::vector<int> labels_;                    ///< 1D array of labels [samples]
    };
}


#endif //MLCPP_DATASET_H
