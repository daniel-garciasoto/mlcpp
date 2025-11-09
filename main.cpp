#include <iostream>

#include "include/core/Dataset.h"
#include "include/supervised/KNN.h"
#include <iostream>

using namespace std;

int main() {
    try {
        cout << "=== KNN Classifier Test ===" << endl;

        // 1. Load dataset
        cout << "Loading dataset..." << endl;
        auto opt_dataset = mlcpp::Dataset::from_csv("data/iris.csv");

        if (!opt_dataset) {
            cerr << "Error: Could not load dataset!" << endl;
            return 1;
        }

        mlcpp::Dataset dataset = *opt_dataset;
        cout << "Dataset loaded: " << dataset.size() << " samples, "
             << dataset.num_features() << " features" << endl;

        // 2. Normalize
        cout << "Normalizing..." << endl;
        dataset.normalize();

        // 3. Split train/test
        cout << "Splitting dataset..." << endl;
        auto [train, test] = dataset.train_test_split(0.2, 42);
        cout << "Train: " << train.size() << " samples" << endl;
        cout << "Test: " << test.size() << " samples" << endl;

        // 4. Train KNN
        cout << "Training KNN..." << endl;
        mlcpp::KNN model(3,mlcpp::manhattan_distance);
        model.fit(train);
        cout << "Model trained with k=" << model.get_k() << endl;

        // 5. Predict one sample
        cout << "Testing single prediction..." << endl;
        const auto& test_features = test.get_features();
        if (!test_features.empty()) {
            int pred = model.predict(test_features[0]);
            cout << "First test sample predicted as: " << pred << endl;
        }

        // 6. Evaluate
        cout << "Evaluating model..." << endl;
        double accuracy = model.score(test);
        cout << "Accuracy: " << (accuracy * 100) << "%" << endl;

        cout << "=== Test Complete ===" << endl;
        return 0;

    } catch (const exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}