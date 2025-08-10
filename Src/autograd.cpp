#include "autograd.h"
#include "function.h"
#include <vector>
#include <unordered_set>

static void build_topo(std::shared_ptr<Tensor> t,
                       std::vector<std::shared_ptr<Tensor>> &topo,
                       std::unordered_set<Tensor*> &visited) {
    if (!t) return;
    if (visited.find(t.get()) != visited.end()) return;
    visited.insert(t.get());

    if (t->creator) {
        for (auto &inp : t->creator->inputs) {
            build_topo(inp, topo, visited);
        }
    }
    topo.push_back(t);
}

void zero_grad_graph(std::shared_ptr<Tensor> t) {
    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;
    build_topo(t, topo, visited);

    for (auto &node : topo) {
        node->zero_grad();
    }
}

void backward(std::shared_ptr<Tensor> t, double grad) {
    if (!t) return;

    // Build topo order (inputs before outputs)
    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;
    build_topo(t, topo, visited);

    // Zero all grads in the graph to avoid accumulation from prior steps
    for (auto &node : topo) {
        node->grad = 0.0;
    }

    // Seed the output gradient
    topo.back()->grad = grad; // topo.back() should be 't'

    // Walk reversed topo (outputs -> inputs)
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        auto node = *it;
        if (!node->creator) continue;
        // The creator expects the output tensor's grad to be set in node->grad
        node->creator->backward(*node);
        // creator->backward should add to inputs' grad (using +=)
    }
}
