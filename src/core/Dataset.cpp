//
// Created by danie on 06/11/2025.
//

#include "../../include/core/Dataset.h"


namespace mlcpp {
    Dataset::Dataset(vector<vector<double> > features, vector<int> labels) {
        this->features_ = features;
        this->labels_ = labels;
    }

    static optional<Dataset> from_csv(const string &filepath,
                                      bool has_header = true,
                                      int label_column = -1) {
        // 1. Verify if it ends with .csv
        if (filepath.size() < 4 ||
            filepath.substr(filepath.size() - 4) != ".csv") {
            return {};
        }

        // 2. Open the file
        ifstream file(filepath);
        if (!file.is_open()) {
            return {};
        }

        // 3. Jump the line if it has a header
        string line;
        if (has_header) {
            getline(file, line);
        }

        // 4. Read data
        vector<vector<double> > features;
        vector<int> labels;

        while (getline(file, line)) {
            istringstream line_stream(line);
            string cell;
            vector<double> row; // Temporal row for features
            int col_index = 0;
            int num_cols = 0;

            // Count columns (only first row)
            if (features.empty()) {
                istringstream temp_stream(line);
                while (getline(temp_stream, cell, ',')) {
                    num_cols++;
                }
                line_stream.clear();
                line_stream.str(line); // Reset stream
            }

            // Determinate label's indexes
            int actual_label_col = label_column;
            if (label_column == -1) {
                actual_label_col = num_cols - 1; // Last column
            }

            int label = 0;
            bool label_found = false;

            // Leer cada celda
            while (getline(line_stream, cell, ',')) {
                try {
                    double value = stod(cell); // String to double

                    if (col_index == actual_label_col) {
                        // It's the label's column
                        label = static_cast<int>(value);
                        label_found = true;
                    } else {
                        // It's a feature
                        row.push_back(value);
                    }
                } catch (const invalid_argument &e) {
                    // Error
                    return {};
                }
                col_index++;
            }

            // Save the row and the label
            if (label_found) {
                features.push_back(row);
                labels.push_back(label);
            }
        }

        file.close();

        // 5. Verify that the data is read
        if (features.empty()) {
            return {};
        }

        // 6. Create and return the Dataset
        return Dataset(features, labels);
    }
}
