#ifndef SOFTMAX_LAYER_H_
#define SOFTMAX_LAYER_H_

#include "Eigen/Dense"

namespace saicdl {

class SoftmaxLayer{
public:
    void softmaxFunction(const Eigen::MatrixXf& inputs,
                          Eigen::MatrixXf& softmax);

    void softmaxLossForwardBackward(const Eigen::MatrixXf& inputs,
                                    const Eigen::VectorXf& label,
                                    Eigen::MatrixXf& d_inputs,
                                    float& loss);
    void test();

}; // end of class SoftmaxLayer

}  // end of namespace saicdl
#endif  // end of define SOFTMAX_LAYER_H_
