//
// Created by danie on 10/11/2025.
//

#ifndef MLCPP_LINEARREGRESSION_H
#define MLCPP_LINEARREGRESSION_H

#include "../core/Dataset.h"

namespace mlcpp {
    /**
     * @brief Linear Regression model for supervised learning.
     *
     * Fits a linear model to predict continuous target values using the
     * Ordinary Least Squares (OLS) method or Gradient Descent.
     *
     * Model equation: y = w₀ + w₁*x₁ + w₂*x₂ + ... + wₙ*xₙ
     * where w₀ is the intercept (bias) and w₁...wₙ are the coefficients (weights).
     */
    class LinearRegression {
    public:
        /**
         * @brief Constructs a Linear Regression model with specified method.
         *
         * @param learning_rate Learning rate for gradient descent (default: 0.01)
         * @param n_iterations Number of iterations for gradient descent (default: 1000)
         * @param method Training method: "normal" for Normal Equation, "gradient" for Gradient Descent (default: "normal")
         *
         * @note Normal Equation is faster for small datasets (< 10,000 samples)
         * @note Gradient Descent is better for large datasets
         *
         * Example usage:
         * @code
         * LinearRegression model1;                           // Normal Equation (default)
         * LinearRegression model2(0.01, 1000, "gradient");   // Gradient Descent
         * @endcode
         */
        explicit LinearRegression(double learning_rate = 0.01,
                                 int n_iterations = 1000,
                                 const std::string& method = "normal");

        /**
         * @brief Trains the linear regression model.
         *
         * Fits the model to the training data using either Normal Equation
         * or Gradient Descent depending on the method specified in constructor.
         *
         * @param X_train Training features [samples][features]
         * @param y_train Training target values [samples]
         *
         * @note Features should be normalized/standardized for best results with Gradient Descent
         * @note Time complexity: O(n * d²) for Normal Equation, O(iterations * n * d) for Gradient Descent
         *
         * Example usage:
         * @code
         * LinearRegression model;
         * model.fit(X_train, y_train);
         * @endcode
         */
        void fit(const std::vector<std::vector<double>>& X_train,
                const std::vector<double>& y_train);

        /**
         * @brief Predicts target value for a single sample.
         *
         * @param sample Feature vector of the sample
         * @return Predicted continuous value
         *
         * @note The model must be trained before prediction
         * @note Time complexity: O(d) where d is the number of features
         *
         * Example usage:
         * @code
         * vector<double> sample = {1500, 3, 2};  // [sq_feet, bedrooms, bathrooms]
         * double price = model.predict(sample);   // Returns predicted house price
         * @endcode
         */
        double predict(const std::vector<double>& sample) const;

        /**
         * @brief Predicts target values for multiple samples.
         *
         * @param X_test Test features [samples][features]
         * @return Vector of predicted values
         *
         * Example usage:
         * @code
         * vector<vector<double>> samples = {{1500, 3, 2}, {2000, 4, 3}};
         * vector<double> predictions = model.predict(samples);
         * @endcode
         */
        std::vector<double> predict(const std::vector<std::vector<double>>& X_test) const;


        /**
         * @brief Gets the model coefficients (weights).
         *
         * @return Vector of weights [w₁, w₂, ..., wₙ]
         */
        const std::vector<double>& get_weights() const { return weights_; }

        /**
         * @brief Gets the bias term.
         *
         * @return Bias value (b)
         */
        double get_bias() const { return bias_; }

    private:
        double alpha_;        ///< Learning rate for gradient descent
        int epochs_;            ///< Number of iterations for gradient descent
        std::string method_;          ///< Training method: "normal" or "gradient"
        std::vector<double> weights_; ///< Model coefficients [features]
        double bias_;            ///< Bias term

        /**
         * @brief Trains using the Normal Equation (closed-form solution).
         *
         * Calculates weights using: w = (X^T * X)^-1 * X^T * y
         *
         * @param X Training features with intercept column
         * @param y Training targets
         *
         * @note Time complexity: O(n * d² + d³) where n = samples, d = features
         * @note Requires matrix inversion - may be numerically unstable
         */
        void fit_normal_equation(const std::vector<std::vector<double>>& X,
                                const std::vector<double>& y);

        /**
         * @brief Trains using Gradient Descent optimization.
         *
         * Iteratively updates weights to minimize MSE loss function.
         *
         * @param X Training features with intercept column
         * @param y Training targets
         *
         * @note Time complexity: O(iterations * n * d)
         * @note Requires careful tuning of learning_rate and n_iterations
         */
        void fit_gradient_descent(const std::vector<std::vector<double>>& X,
                                 const std::vector<double>& y);


    };
}


#endif //MLCPP_LINEARREGRESSION_H