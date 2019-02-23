#include "activation_layer.hpp"

namespace saicdl {

// top = max(bottom, 0)
void ActivationLayer::reluForward(const Eigen::MatrixXf& bottom,
                                  Eigen::MatrixXf& top) {
    // an expression of the coefficient-wise max of *this and the scalar other
    top = bottom.cwiseMax(0);
}

// if x > 0, dx = drelu, else dx = 0
void ActivationLayer::reluBackward(const Eigen::MatrixXf& top,
                                   const Eigen::MatrixXf& d_top,
                                   Eigen::MatrixXf& d_bottom) {
    d_bottom = (top.array() <= 0).select(0, d_top);
}

// y = 1 / (1 + exp(-x))
void ActivationLayer::sigmoidForward(const Eigen::MatrixXf& bottom,
                                     Eigen::MatrixXf& top) {
    top = 1 / (1 + (-bottom).array().exp());
}

// dx = y ( 1 - y)
void ActivationLayer::sigmoidBackward(const Eigen::MatrixXf& top,
                                      const Eigen::MatrixXf& d_top,
                                      Eigen::MatrixXf& d_bottom) {
    d_bottom = d_top.array() * top.array() * (1 - top.array());
}

}   // end of namespace saicdl
