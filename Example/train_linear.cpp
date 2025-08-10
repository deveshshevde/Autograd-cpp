#include <iostream>
#include "tensor.h"
#include "ops.h"
#include "autograd.h"

int main() {
    auto w = std::make_shared<Tensor>(0.0);
    auto b = std::make_shared<Tensor>(0.0);
    double lr = 0.01;

    for (int epoch = 0; epoch < 100; ++epoch) {
        auto x = std::make_shared<Tensor>(2.0);
        auto y_true = std::make_shared<Tensor>(7.0);

        auto y_pred = add(mul(w, x), b);
        auto diff = add(y_pred, std::make_shared<Tensor>(-y_true->value));
        auto loss = mul(diff, diff);

        w->zero_grad();
        b->zero_grad();
        backward(loss);

        w->value -= lr * w->grad;
        b->value -= lr * b->grad;

        std::cout << "Epoch " << epoch
                    << " Loss: " << loss->value
                    << " w=" << w->value
                    << " b=" << b->value << "\n";
    }
}
