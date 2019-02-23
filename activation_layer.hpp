#ifndef ACTIVATION_LAYER_H_
#define ACTIVATION_LAYER_H_

#include "Eigen/Dense"

namespace saicdl{

class ActivationLayer {
public:
    void reluForward(const Eigen::MatrixXf& bottom, Eigen::MatrixXf& top );
    void reluBackward(const Eigen::MatrixXf& top, const Eigen::MatrixXf& d_top, Eigen::MatrixXf& d_bottom);
    void sigmoidForward(const Eigen::MatrixXf& bottom, Eigen::MatrixXf& top);
    void sigmoidBackward(const Eigen::MatrixXf& top, const Eigen::MatrixXf& d_top, Eigen::MatrixXf& d_bottom);
};  // end of class ActivationLayer

}   // end of namespace 

#endif  // end of ifndef

