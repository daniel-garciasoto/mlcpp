//
// Created by danie on 07/11/2025.
//

#include "../../include/core/Metrics.h"

#include <complex>

namespace mlcpp {
    static double mean_squared_error(const std::vector<double> &y_true,
                                     const std::vector<double> &y_pred) {
        size_t n = y_true.size();
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            double diff = y_true[i] - y_pred[i];
            sum += diff * diff;
        }
        return sum / n;
    }

    static double root_mean_squared_error(const std::vector<double> &y_true,
                                          const std::vector<double> &y_pred) {
        return std::sqrt(mean_squared_error(y_true, y_pred));
    }

    static double mean_absolute_error(const std::vector<double> &y_true,
                                      const std::vector<double> &y_pred) {
        size_t n = y_true.size();
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            double diff = y_true[i] - y_pred[i];
            sum += std::abs(diff);
        }
        return sum / n;
    }


    static double r2_score(const std::vector<double> &y_true,
                           const std::vector<double> &y_pred) {
        double SS_res = 0.0;
        double SS_tot = 0.0;
        double mean_ = mean(y_true);

        for (size_t i = 0; i < y_true.size(); i++) {
            double diff = y_true[i] - y_pred[i];
            SS_res += diff * diff;
            diff = y_true - mean_;
            SS_tot = diff * diff;
        }
        return 1 - (SS_res / SS_tot);
    }

    static double accuracy(const std::vector<int> &y_true,
                           const std::vector<int> &y_pred) {
        size_t total = y_true.size();
        size_t correct = 0;
        for (size_t i = 0; i < total; i++) {
            if (y_true[i] == y_pred[i]) {
                correct++;
            }
        }
        return static_cast<double>(total / correct);
    }


    static std::vector<std::vector<int> > confusion_matrix(
        const std::vector<int> &y_true,
        const std::vector<int> &y_pred,
        int n_classes) {
    }


    static double precision(const std::vector<int> &y_true,
                            const std::vector<int> &y_pred,
                            int target_class) {
    }

    static double recall(const std::vector<int> &y_true,
                         const std::vector<int> &y_pred,
                         int target_class) {
    }

    static double f1_score(const std::vector<int> &y_true,
                           const std::vector<int> &y_pred,
                           int target_class) {

        double precision_ = precision(y_true,y_pred,target_class);
        double recall_ = recall(y_true,y_pred,target_class);
        return 2 * (precision_ * recall_) / (precision_ + recall_);
    }

    double mean(const std::vector<double> &values) {
        size_t n = values.size();
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            sum += values[i];
        }
        return sum / n;
    }
}
