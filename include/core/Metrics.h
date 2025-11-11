//
// Created by danie on 07/11/2025.
//

#ifndef MLCPP_METRICS_H
#define MLCPP_METRICS_H
#include <vector>

namespace mlcpp {
    /**
     * @brief Collection of evaluation metrics for machine learning models.
     *
     * Provides metrics for both regression and classification tasks.
     * All functions are designed to be stateless and reusable.
     */
    class Metrics {
    public:
        // ==================== REGRESSION METRICS ====================

        /**
         * @brief Calculates Mean Squared Error (MSE).
         *
         * MSE = (1/n) * Σ(y_true - y_pred)²
         *
         * @param y_true True target values
         * @param y_pred Predicted target values
         * @return Mean Squared Error
         *
         * @note Lower is better (0 = perfect predictions)
         * @note Sensitive to outliers (squared errors)
         * @note Time complexity: O(n)
         *
         * Example usage:
         * @code
         * vector<double> y_true = {100, 200, 300};
         * vector<double> y_pred = {110, 190, 310};
         * double mse = Metrics::mean_squared_error(y_true, y_pred);  // 66.67
         * @endcode
         */
        static double mean_squared_error(const std::vector<double>& y_true,
                                        const std::vector<double>& y_pred);

        /**
         * @brief Calculates Root Mean Squared Error (RMSE).
         *
         * RMSE = sqrt(MSE)
         *
         * @param y_true True target values
         * @param y_pred Predicted target values
         * @return Root Mean Squared Error
         *
         * @note Same units as the target variable
         * @note Easier to interpret than MSE
         * @note Time complexity: O(n)
         */
        static double root_mean_squared_error(const std::vector<double>& y_true,
                                             const std::vector<double>& y_pred);

        /**
         * @brief Calculates Mean Absolute Error (MAE).
         *
         * MAE = (1/n) * Σ|y_true - y_pred|
         *
         * @param y_true True target values
         * @param y_pred Predicted target values
         * @return Mean Absolute Error
         *
         * @note Less sensitive to outliers than MSE
         * @note Same units as the target variable
         * @note Time complexity: O(n)
         */
        static double mean_absolute_error(const std::vector<double>& y_true,
                                         const std::vector<double>& y_pred);

        /**
         * @brief Calculates R² (coefficient of determination) score.
         *
         * R² = 1 - (SS_res / SS_tot)
         * where SS_res = Σ(y_true - y_pred)²
         *       SS_tot = Σ(y_true - mean(y_true))²
         *
         * @param y_true True target values
         * @param y_pred Predicted target values
         * @return R² score
         *
         * @note 1.0 = perfect prediction
         * @note 0.0 = model predicts mean value (baseline)
         * @note Negative = worse than predicting mean
         * @note Time complexity: O(n)
         *
         * Example usage:
         * @code
         * double r2 = Metrics::r2_score(y_true, y_pred);
         * cout << "R²: " << r2 << endl;  // "R²: 0.85"
         * @endcode
         */
        static double r2_score(const std::vector<double>& y_true,
                              const std::vector<double>& y_pred);

        // ==================== CLASSIFICATION METRICS ====================

        /**
         * @brief Calculates classification accuracy.
         *
         * Accuracy = (correct predictions) / (total predictions)
         *
         * @param y_true True class labels
         * @param y_pred Predicted class labels
         * @return Accuracy between 0.0 and 1.0
         *
         * @note Time complexity: O(n)
         *
         * Example usage:
         * @code
         * vector<int> y_true = {0, 1, 2, 1, 0};
         * vector<int> y_pred = {0, 1, 2, 2, 0};
         * double acc = Metrics::accuracy(y_true, y_pred);  // 0.8 (4/5 correct)
         * @endcode
         */
        static double accuracy(const std::vector<int>& y_true,
                              const std::vector<int>& y_pred);

        /**
         * @brief Generates a confusion matrix for classification.
         *
         * Confusion matrix shows true vs predicted class counts.
         * Matrix[i][j] = count where true class is i and predicted is j.
         *
         * @param y_true True class labels
         * @param y_pred Predicted class labels
         * @param n_classes Number of classes (default: auto-detect)
         * @return Confusion matrix [n_classes][n_classes]
         *
         * @note Time complexity: O(n)
         *
         * Example usage:
         * @code
         * auto cm = Metrics::confusion_matrix(y_true, y_pred);
         * // For binary classification:
         * // [[TN, FP],
         * //  [FN, TP]]
         * @endcode
         */
        static std::vector<std::vector<int>> confusion_matrix(
            const std::vector<int>& y_true,
            const std::vector<int>& y_pred,
            int n_classes = -1);

        /**
         * @brief Calculates precision for a specific class.
         *
         * Precision = TP / (TP + FP)
         *
         * @param y_true True class labels
         * @param y_pred Predicted class labels
         * @param target_class Class to calculate precision for
         * @return Precision between 0.0 and 1.0
         *
         * @note Precision = "Of all predicted positives, how many were correct?"
         * @note Returns 0.0 if no predictions for this class
         */
        static double precision(const std::vector<int>& y_true,
                               const std::vector<int>& y_pred,
                               int target_class);

        /**
         * @brief Calculates recall for a specific class.
         *
         * Recall = TP / (TP + FN)
         *
         * @param y_true True class labels
         * @param y_pred Predicted class labels
         * @param target_class Class to calculate recall for
         * @return Recall between 0.0 and 1.0
         *
         * @note Recall = "Of all actual positives, how many were found?"
         * @note Returns 0.0 if no actual samples of this class
         */
        static double recall(const std::vector<int>& y_true,
                            const std::vector<int>& y_pred,
                            int target_class);

        /**
         * @brief Calculates F1 score for a specific class.
         *
         * F1 = 2 * (Precision * Recall) / (Precision + Recall)
         *
         * @param y_true True class labels
         * @param y_pred Predicted class labels
         * @param target_class Class to calculate F1 score for
         * @return F1 score between 0.0 and 1.0
         *
         * @note F1 is the harmonic mean of precision and recall
         */
        static double f1_score(const std::vector<int>& y_true,
                              const std::vector<int>& y_pred,
                              int target_class);

    private:
        /**
         * @brief Calculates the mean of a vector.
         *
         * @param values Vector of values
         * @return Mean value
         */
        static double mean(const std::vector<double>& values);
    };
}


#endif //MLCPP_METRICS_H