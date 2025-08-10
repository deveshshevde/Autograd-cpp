#pragma once
#include <vector>
#include <memory>
#include "tensor.h"

struct Function {
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::shared_ptr<Tensor> output;

    virtual void backward(Tensor &out_grad) = 0;
    virtual ~Function() = default;
};
