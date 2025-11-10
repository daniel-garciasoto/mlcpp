//
// Created by danie on 06/11/2025.
//

#ifndef MLCPP_KNN_H
#define MLCPP_KNN_H
#include <vector>
#include "../core/Dataset.h"
#include "../core/Distance.h"

namespace mlcpp {
    /**
     * @brief K-Nearest Neighbors (KNN) classifier for supervised learning.
     *
     * A non-parametric, lazy learning algorithm that classifies samples based on
     * the majority vote of their k nearest neighbors in the feature space.
     *
     * @note This is a "lazy" algorithm - training only stores the data,
     *       all computation happens during prediction.
     */
    class KNN {
    public:
        /**
         * @brief Constructs a KNN classifier with specified parameters.
         *
         * @param k Number of nearest neighbors to consider (default: 3)
         * @param distance Distance metric function to use (default: euclidean_distance)
         *
         * @note k should be odd to avoid ties in binary classification
         * @note Larger k values make the model more robust but less sensitive to local patterns
         *
         * Example usage:
         * @code
         * KNN model1(5);                              // k=5, Euclidean distance
         * KNN model2(3, manhattan_distance);          // k=3, Manhattan distance
         * @endcode
         */
        explicit KNN(int k = 3, DistanceMetric distance = euclidean_distance);

        /**
         * @brief Trains the KNN model by storing the training data.
         *
         * Since KNN is a lazy learner, this method simply stores the features
         * and labels from the dataset for later use during prediction.
         *
         * @param dataset Training dataset containing features and labels
         *
         * @note Time complexity: O(1) - just stores references to the data
         * @note Any previous training data is overwritten
         *
         * Example usage:
         * @code
         * KNN model(3);
         * model.fit(train_dataset);
         * @endcode
         */
        void fit(Dataset& dataset);

        /**
         * @brief Predicts the class label for a single sample.
         *
         * Finds the k nearest neighbors in the training set and returns
         * the most common label among them (majority vote).
         *
         * @param sample Feature vector of the sample to classify
         * @return Predicted class label (integer)
         *
         * @note Time complexity: O(n * d) where n = training samples, d = features
         * @note The model must be trained (fit) before calling this
         *
         * Example usage:
         * @code
         * vector<double> sample = {5.1, 3.5, 1.4, 0.2};
         * int label = model.predict(sample);  // Returns 0, 1, or 2
         * @endcode
         */
        int predict(const std::vector<double>& sample) const;

        /**
         * @brief Predicts class labels for multiple samples.
         *
         * Applies the predict method to each sample in the input.
         *
         * @param samples 2D vector where each row is a sample to classify
         * @return Vector of predicted labels, one for each input sample
         *
         * @note Time complexity: O(m * n * d) where m = test samples,
         *                        n = training samples, d = features
         *
         * Example usage:
         * @code
         * vector<vector<double>> samples = {{5.1, 3.5, 1.4, 0.2},
         *                                   {6.3, 2.9, 5.6, 1.8}};
         * vector<int> predictions = model.predict(samples);  // {0, 2}
         * @endcode
         */
        std::vector<int> predict(const std::vector<std::vector<double>>& samples) const;

        /**
         * @brief Calculates the accuracy of the model on a test dataset.
         *
         * Compares predictions against true labels and returns the proportion
         * of correct predictions.
         *
         * @param test_dataset Dataset containing test samples and their true labels
         * @return Accuracy as a value between 0.0 (0%) and 1.0 (100%)
         *
         * @note The model must be trained before evaluation
         * @note Time complexity: O(m * n * d) where m = test samples
         *
         * Example usage:
         * @code
         * double accuracy = model.score(test_dataset);
         * cout << "Accuracy: " << (accuracy * 100) << "%" << endl;  // "Accuracy: 96.7%"
         * @endcode
         */
        double score(const Dataset& test_dataset) const;

        /**
         * @brief Gets the number of neighbors (k) used by the classifier.
         *
         * @return The k value
         */
        int get_k() const { return k_; }

    private:
        int k_;                                      ///< Number of nearest neighbors to consider
        DistanceMetric distance_;                    ///< Distance metric function
        std::vector<std::vector<double>> X_train_;   ///< Training features [samples][features]
        std::vector<int> y_train_;                   ///< Training labels [samples]

        /**
         * @brief Finds the indices of the k nearest neighbors for a given sample.
         *
         * Calculates distances from the sample to all training samples,
         * then returns the indices of the k closest ones.
         *
         * @param sample Feature vector to find neighbors for
         * @return Vector of indices pointing to the k nearest training samples
         *
         * @note Time complexity: O(n log k) using partial_sort
         */
        std::vector<size_t> find_k_nearest(const std::vector<double>& sample) const;

        /**
         * @brief Determines the predicted label by majority vote among neighbors.
         *
         * Counts the occurrence of each label among the given neighbor indices
         * and returns the most frequent one.
         *
         * @param neighbor_indices Indices of the nearest neighbors
         * @return The most common label among the neighbors
         *
         * @note In case of a tie, returns the label that appears first
         *       (implementation dependent on map ordering)
         * @note Time complexity: O(k) where k is the number of neighbors
         */
        int majority_vote(const std::vector<size_t>& neighbor_indices) const;
    };
}

#endif //MLCPP_KNN_H