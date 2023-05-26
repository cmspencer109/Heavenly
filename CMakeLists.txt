cmake_minimum_required(VERSION 3.16)
project(Heavenly)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add executable and link OpenCV libraries
add_executable(Heavenly main.cpp)
target_link_libraries(Heavenly ${OpenCV_LIBS})
