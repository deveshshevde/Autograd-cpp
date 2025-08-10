#pragma once
#include <memory>

struct Function; 

struct Tensor {
    double value;
    double grad = 0.0;
    std::shared_ptr<Function> creator;

    Tensor(double v);
    void zero_grad(); 
};
