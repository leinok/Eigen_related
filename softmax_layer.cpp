#include <iostream>
#include "softmax_layer.hpp"

namespace saicdl{

void SoftmaxLayer::softmaxFunction(const Eigen::MatrixXf& inputs,
                                   Eigen::MatrixXf& softmax) {
    softmax = inputs.array().exp();
    Eigen::VectorXf softmax_rowsum = softmax.rowwise().sum();
    softmax = softmax.array().colwise() / softmax_rowsum.array();
}

void SoftmaxLayer::softmaxLossForwardBackward(const Eigen::MatrixXf& inputs,
                                              const Eigen::VectorXf& label,
                                              Eigen::MatrixXf& d_inputs,
                                              float& loss) {
    Eigen::MatrixXf softmax;
    softmaxFunction(inputs, softmax);

    Eigen::MatrixXf real_label = Eigen::MatrixXf::Zero(softmax.rows(), softmax.cols());
    assert(inputs.rows() == label.rows());

    for (int i = 0; i < real_label.rows(); ++i) {
        real_label(i, label(i)) = 1;
    }   

    loss -= (real_label.array() * softmax.array().log()).mean();
    d_inputs = (softmax - real_label) / (inputs.rows() * inputs.cols());
        
}

void SoftmaxLayer::test() {
    std::cout << "Test..." << std::endl;
}

} // end of namespace saicdl
