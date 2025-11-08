#include <iostream>

#include "include/core/Dataset.h"
#include "include/supervised/KNN.h"


int main() {
    auto dataset = mlcpp::Dataset::from_csv("data/iris.csv");
    dataset->normalize();

    auto [train, test] = dataset->train_test_split(0.2);

    mlcpp::KNN model(5);
    model.fit(train);

    // 1. Predict a sample
    vector<double> single_sample = {5.1, 3.5, 1.4, 0.2};
    int label = model.predict(single_sample);
    cout << "Predicted label: " << label << endl;

    // 2. Predict multiple samples
    vector<int> predictions = model.predict(test.get_features());
    cout << "Predictions: ";
    for (int pred : predictions) {
        cout << pred << " ";
    }
    cout << endl;

    // 3. Evaluate accuracy
    double acc = model.score(test);
    cout << "Accuracy: " << acc * 100 << "%" << endl;

    return 0;
}