//
// Created by danie on 06/11/2025.
//

#ifndef MLCPP_DISTANCE_H
#define MLCPP_DISTANCE_H
#include <vector>
#include <cmath>
#include <functional>
using namespace std;

namespace mlcpp {
    using DistanceMetric = function<double(const vector<double> &, const vector<double> &)>;

    inline double euclidean_distance(const vector<double> &a, const vector<double> &b) {
        double sum = 0.0;
        for (int i = 0; i < a.size(); i++) {
            sum += pow(a[i] - b[i], 2);
        }
        return sqrt(sum);
    }

    inline double manhattan_distance(const vector<double> &a, const vector<double> &b) {
        double sum = 0.0;
        for (int i = 0; i < a.size(); i++) {
            sum += abs(a[i] - b[i]);
        }

        return sum;
    }
}


#endif //MLCPP_DISTANCE_H
