//
// Created by danie on 10/11/2025.
//

#include "../../include/supervised/LinearRegression.h"
using namespace std;

namespace mlcpp {
    LinearRegression::LinearRegression(double learning_rate, int epochs, const std::string &method) {
        this->alpha_ = learning_rate;
        this->epochs_ = epochs;
        this->method_ = method;
    }

    void LinearRegression::fit(const std::vector<std::vector<double> > &X_train, const std::vector<double> &y_train) {
        if (this->method_ == "gradient") {
            fit_gradient_descent(X_train,y_train);
        } else if (this->method_ == "normal") {
            fit_normal_equation(X_train,y_train);
        }
    }


    void LinearRegression::fit_gradient_descent(const std::vector<std::vector<double> > &X,
                                                const std::vector<double> &y) {
        // number of datapoints
        double m = X.size();

        // number of features
        double n = X[0].size();

        // initializating bias and weight at 0
        double b = 0.0;
        vector<double> w(n, 0.0);

        for (size_t e = 0; e < this->epochs_; e++) {
            double b_updated = 0.0;
            vector<double> w_updated(n, 0.0);

            for (size_t i = 0; i < m; i++) {
                double pred = b;
                for (size_t j = 0; j < n; j++) {
                    pred += X[i][j] * w[i] + b;
                }
                double error = y[i] - pred;
                for (size_t j = 0; j < n; j++) {
                    w_updated[j] += (-1) * X[i][j] * error / m;
                }
                b_updated += (-1) * error / m;
            }

            // Weights and bias update
            for (size_t i = 0; i < n; i++) {
                w[i] = w[i] - w_updated[i] * this->alpha_;
            }
            b += b_updated - this->alpha_;
        }
        //Class weights and bias update
        this->weights_ = w;
        this->bias_ = b;
    }
}
