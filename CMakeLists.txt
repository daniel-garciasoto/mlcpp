cmake_minimum_required(VERSION 4.0)
project(mlcpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(mlcpp main.cpp
        src/supervised/KNN.cpp
        include/supervised/KNN.h
        include/core/Distance.h
        src/core/Dataset.cpp
        include/core/Dataset.h
        src/core/Metrics.cpp
        include/core/Metrics.h
        Doxyfile
        src/supervised/LinearRegression.cpp
        include/supervised/LinearRegression.h
)
