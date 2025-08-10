#include "ops.h"


struct Add : public Function {
    void backward(Tensor &out_grad) override {
        inputs[0]->grad += out_grad.grad;
        inputs[1]->grad += out_grad.grad;
    }
};

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    auto op = std::make_shared<Add>();
    op->inputs = {a, b};
    auto out = std::make_shared<Tensor>(a->value + b->value);
    out->creator = op;
    op->output = out;
    return out;
}

// Multiplication
struct Mul : public Function {
    void backward(Tensor &out_grad) override {
        inputs[0]->grad += out_grad.grad * inputs[1]->value;
        inputs[1]->grad += out_grad.grad * inputs[0]->value;
    }
};

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    auto op = std::make_shared<Mul>();
    op->inputs = {a, b};
    auto out = std::make_shared<Tensor>(a->value * b->value);
    out->creator = op;
    op->output = out;
    return out;
}
