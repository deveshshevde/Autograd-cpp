#include "tensor.h"

Tensor::Tensor(double v) : value(v) {}

void Tensor::zero_grad() {
    grad = 0.0;
}
