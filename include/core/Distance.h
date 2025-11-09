//
// Created by danie on 06/11/2025.
//

#ifndef MLCPP_DISTANCE_H
#define MLCPP_DISTANCE_H
#include <vector>
#include <cmath>
#include <functional>
namespace mlcpp {
    /**
     * @brief Type alias for distance metric functions.
     *
     * A distance metric takes two feature vectors and returns a scalar distance value.
     * All distance metrics in this library follow this signature.
     *
     * @note Lower values indicate more similar vectors
     * @note Distance functions should satisfy: d(a,b) >= 0 and d(a,a) = 0
     */
    using DistanceMetric = std::function<double(const std::vector<double>&, const std::vector<double>&)>;

    /**
     * @brief Calculates the Euclidean (L2) distance between two vectors.
     *
     * The Euclidean distance is the straight-line distance between two points
     * in n-dimensional space. Also known as L2 norm or L2 distance.
     *
     * Formula: sqrt(sum((a_i - b_i)^2))
     *
     * @param a First feature vector
     * @param b Second feature vector
     * @return Euclidean distance between a and b
     *
     * @note Both vectors must have the same dimensionality
     * @note This is the most commonly used distance metric
     * @note Sensitive to feature scaling - normalize data for best results
     * @note Time complexity: O(d) where d is the number of dimensions
     *
     * Example usage:
     * @code
     * vector<double> point1 = {1.0, 2.0, 3.0};
     * vector<double> point2 = {4.0, 5.0, 6.0};
     * double dist = euclidean_distance(point1, point2);  // Returns ~5.196
     * @endcode
     */
    inline double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

    /**
     * @brief Calculates the Manhattan (L1) distance between two vectors.
     *
     * The Manhattan distance is the sum of absolute differences between coordinates.
     * Also known as L1 norm, taxicab distance, or city block distance.
     *
     * Formula: sum(|a_i - b_i|)
     *
     * @param a First feature vector
     * @param b Second feature vector
     * @return Manhattan distance between a and b
     *
     * @note Both vectors must have the same dimensionality
     * @note Less sensitive to outliers than Euclidean distance
     * @note Faster to compute (no square root operation)
     * @note Works well in high-dimensional spaces
     * @note Time complexity: O(d) where d is the number of dimensions
     *
     * Example usage:
     * @code
     * vector<double> point1 = {1.0, 2.0, 3.0};
     * vector<double> point2 = {4.0, 5.0, 6.0};
     * double dist = manhattan_distance(point1, point2);  // Returns 9.0
     * @endcode
     */
    inline double manhattan_distance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            sum += abs(a[i] - b[i]);
        }
        return sum;
    }

    /**
     * @brief Calculates the Chebyshev (L∞) distance between two vectors.
     *
     * The Chebyshev distance is the maximum absolute difference across all dimensions.
     * Also known as L∞ norm or maximum metric.
     *
     * Formula: max(|a_i - b_i|)
     *
     * @param a First feature vector
     * @param b Second feature vector
     * @return Chebyshev distance between a and b
     *
     * @note Both vectors must have the same dimensionality
     * @note Useful when you care about the worst-case difference
     * @note Very fast to compute
     * @note Time complexity: O(d) where d is the number of dimensions
     *
     * Example usage:
     * @code
     * vector<double> point1 = {1.0, 2.0, 3.0};
     * vector<double> point2 = {4.0, 5.0, 6.0};
     * double dist = chebyshev_distance(point1, point2);  // Returns 3.0
     * @endcode
     */
    inline double chebyshev_distance(const std::vector<double>& a, const std::vector<double>& b) {
        double max_diff = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            double diff = abs(a[i] - b[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        return max_diff;
    }

    /**
     * @brief Calculates the Minkowski distance between two vectors.
     *
     * The Minkowski distance is a generalization of both Euclidean and Manhattan distances.
     *
     * Formula: (sum(|a_i - b_i|^p))^(1/p)
     *
     * Special cases:
     * - p = 1: Manhattan distance
     * - p = 2: Euclidean distance
     * - p → ∞: Chebyshev distance
     *
     * @param a First feature vector
     * @param b Second feature vector
     * @param p The order of the Minkowski distance (default: 2.0)
     * @return Minkowski distance between a and b
     *
     * @note Both vectors must have the same dimensionality
     * @note p must be >= 1
     * @note Higher p values give more weight to large differences
     * @note Time complexity: O(d) where d is the number of dimensions
     *
     * Example usage:
     * @code
     * vector<double> point1 = {1.0, 2.0, 3.0};
     * vector<double> point2 = {4.0, 5.0, 6.0};
     * double dist1 = minkowski_distance(point1, point2, 1.0);  // Manhattan: 9.0
     * double dist2 = minkowski_distance(point1, point2, 2.0);  // Euclidean: ~5.196
     * double dist3 = minkowski_distance(point1, point2, 3.0);  // p=3: ~4.327
     * @endcode
     */
    inline double minkowski_distance(const std::vector<double>& a, const std::vector<double>& b, double p = 2.0) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            sum += pow(abs(a[i] - b[i]), p);
        }
        return pow(sum, 1.0 / p);
    }
}


#endif //MLCPP_DISTANCE_H
