cmake_minimum_required(VERSION 3.10)
project(AUTOGRAD_CPP)

set(CMAKE_CXX_STANDARD 17)
    
add_library(AUTOGRAD_CPP SHARED
    Src/tensor.cpp
    Src/ops.cpp
    Src/autograd.cpp
)

target_include_directories(AUTOGRAD_CPP PUBLIC Include)

add_executable(train_linear Example/train_linear.cpp)
target_link_libraries(train_linear AUTOGRAD_CPP)
