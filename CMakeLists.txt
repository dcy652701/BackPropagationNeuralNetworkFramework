cmake_minimum_required(VERSION 3.7)

project(DeepLearning_framework)

set(CMAKE_BUILD_TYPE release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -std=c++14")

add_executable(
  main
  src/main.cpp
  src/Neuron.cpp
  src/Matrix.cpp
  src/Layer.cpp
  src/Matrix_Math.cpp
  src/NeuralNetwork.cpp
  src/Fetch_Dataset.cpp
)
