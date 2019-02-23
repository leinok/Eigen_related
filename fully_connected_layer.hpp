#ifndef FULLY_CONNECTED_LAYER_H_
#define FULLY_CONNECTED_LAYER_H_

#include "Eigen/Dense"

namespace saicdl{

class FullyConnectedLayer{
public:
    void forward(const Eigen::MatrixXf& inputs,
                 const Eigen::MatrixXf& weights,
                 const Eigen::VectorXf& bias,
                 Eigen::MatrixXf& outputs);

    void backward(const Eigen::MatrixXf& inputs,
                  const Eigen::MatrixXf& weights,
                  const Eigen::VectorXf& bias,
                  const Eigen::MatrixXf& d_outputs,
                  Eigen::MatrixXf& d_inputs,
                  Eigen::MatrixXf& d_weights,
                  Eigen::VectorXf& d_bias);
};

}

#endif  // end of define
