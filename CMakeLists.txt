cmake_minimum_required(VERSION 3.16)
project(Heavenly)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find wxWidgets package
find_package(wxWidgets REQUIRED COMPONENTS core base)

# Include wxWidgets macros
include(${wxWidgets_USE_FILE})

# Add executable and link libraries
add_executable(Heavenly app.cpp)
target_link_libraries(Heavenly ${OpenCV_LIBS} ${wxWidgets_LIBRARIES})
