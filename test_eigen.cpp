#include <iostream>
#include <typeinfo>
#include <string>
#include "softmax_layer.hpp"
#include "fully_connected_layer.hpp"
#include "activation_layer.hpp"

int main() {
        int batch_size{4}, input_size{3}, output_size{2};

        Eigen::MatrixXf inputs(batch_size, input_size);
        inputs << 1, -2, 3, 4, -5, 6, 7, -8, -9, 10, 11, -12;
        Eigen::MatrixXf weights(input_size, output_size);
        weights << .55, -.88, .75, -1.1, -0.11, 0.002;
        Eigen::VectorXf bias(output_size);
        bias << 3, -2;
        Eigen::VectorXf label(batch_size);
        label << 1, 0, 1, 1;

     
        saicdl::FullyConnectedLayer fcl;
        saicdl::ActivationLayer al;
        saicdl::SoftmaxLayer sl;

        Eigen::MatrixXf fc_outputs;
        fcl.forward(inputs, weights, bias, fc_outputs);
        Eigen::MatrixXf relu_outputs;
        al.sigmoidForward(fc_outputs, relu_outputs);

        Eigen::MatrixXf d_inputs, d_fc_weights, d_relu_outputs, d_fc_outputs;

        float loss;

        sl.softmaxLossForwardBackward(relu_outputs, label, d_relu_outputs, loss);

        std::cout << "Loss is: " << loss << '\n';

        al.sigmoidBackward(relu_outputs, d_relu_outputs, d_fc_outputs);

        Eigen::VectorXf d_fc_bias;
        fcl.backward(inputs, weights, bias, d_fc_outputs, d_inputs, d_fc_weights, d_fc_bias);

        std::cout << "relu_outputs: " << relu_outputs << '\n';
        std::cout << "d_relu_outputs: " << d_relu_outputs << '\n';
        std::cout << "d_inputs: " << d_inputs << '\n';
        std::cout << "d_fc_weights: " << d_fc_weights << '\n';
        std::cout << "d_fc_bias: " << d_fc_bias << '\n';
        std::cout << "d_fc_outputs: " << d_fc_outputs << '\n';

     

}

