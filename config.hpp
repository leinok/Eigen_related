#ifndef CONFIG_H_
#define CONFIG_H_

#include <unsupported/Eigen/CXX11/ThreadPool>
#include <unsupported/Eigen/CXX11/Tensor>

namespace saicdl {

typedef Eigen::Tensor<float, 1, Eigen::ColMajor> Tensor1xf;
typedef Eigen::Tensor<float, 2, Eigen::ColMajor> Tensor2xf;
typedef Eigen::Tensor<float, 3, Eigen::ColMajor> Tensor3xf;
typedef Eigen::Tensor<float, 4, Eigen::ColMajor> Tensor4xf;
typedef Eigen::Tensor<float, 5, Eigen::ColMajor> Tensor5xf;

typedef Eigen::Tensor<float, 1, Eigen::RowMajor> Tensor1rf;
typedef Eigen::Tensor<float, 2, Eigen::RowMajor> Tensor2rf;
typedef Eigen::Tensor<float, 3, Eigen::RowMajor> Tensor3rf;
typedef Eigen::Tensor<float, 4, Eigen::RowMajor> Tensor4rf;
typedef Eigen::Tensor<float, 5, Eigen::RowMajor> Tensor5rf;

const int EIGEN_USE_THREADS=8;


}

#endif // end of CONFIG_H_
