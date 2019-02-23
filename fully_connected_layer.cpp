#include "fully_connected_layer.hpp"

namespace saicdl{

// Y = X * W + b
void FullyConnectedLayer::forward(const Eigen::MatrixXf& inputs,
                                  const Eigen::MatrixXf& weights,
                                  const Eigen::VectorXf& bias,
                                  Eigen::MatrixXf& outputs) {
    outputs = inputs * weights;
    outputs.rowwise() += bias.transpose();
}

// dW = x.T * dy, dx = dy * W.T, db = dy
void FullyConnectedLayer::backward(const Eigen::MatrixXf& inputs,
                                   const Eigen::MatrixXf& weights,
                                   const Eigen::VectorXf& bias,
                                   const Eigen::MatrixXf& d_outputs,
                                   Eigen::MatrixXf& d_inputs,
                                   Eigen::MatrixXf& d_weights,
                                   Eigen::VectorXf& d_bias) {
    d_weights = inputs.transpose() * d_outputs;
    d_inputs = d_outputs * weights.transpose();
    d_bias = d_outputs.colwise().sum();
}

}   // end of namespace
