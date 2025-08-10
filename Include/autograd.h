#pragma once
#include <memory>
#include "tensor.h"

void backward(std::shared_ptr<Tensor> t, double grad = 1.0);
void zero_grad_graph(std::shared_ptr<Tensor> t);
