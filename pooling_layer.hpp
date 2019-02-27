#ifndef POOLING_LAYER_H_
#define POOLING_LAYER_H_

#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include <vector>
#include "config.hpp"

namespace saicdl {

enum class PoolingMethod {    // enum identifier
    MAX,    // assigned 0, predix with a standard prefix to prevent naming convention
    AVG,    // assigned 1, trailing comma is allowed in c++11
};

class PoolingLayer {
public:
    PoolingLayer(std::vector<int> pooling_shape, 
                 PoolingMethod pooling_method,
                 int padding = 0);
    ~PoolingLayer() {}
    
    void extractImagePatches(const Tensor4xf& bottom, Tensor5xf& patches);
    Eigen::DSizes<int, 4> getTopShape(const Tensor4xf& bottom);
    void forward(const Tensor4xf& bottom, 
                 Tensor4xf& top,
                 const Eigen::ThreadPoolDevice& device);

    std::vector<int> decode_index(std::vector<int> dim, int index);
    void maxPoolingBackward(const Tensor4xf& bottom, const Tensor4xf& d_top,
                            Tensor4xf& d_bottom);
    void avgPoolingBackward(const Tensor4xf& d_top, Tensor4xf& d_bottom);
    void backward(const Tensor4xf& bottom, const Tensor4xf& d_top, Tensor4xf& d_bottom, 
                  const Eigen::ThreadPoolDevice& device);
    
private:
    int m_k_h_sz;
    int m_k_w_sz;
    int m_h_stride;
    int m_w_stride;

    int m_padding;
    PoolingMethod m_pooling_method;

  };  // End of class PoolingLayer

}   // End of namespace

#endif // end of ifndef
