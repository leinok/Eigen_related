#include "pooling_layer.hpp"
#include "base_functions.hpp"

namespace saicdl {

PoolingLayer::PoolingLayer(std::vector<int> pooling_shape,
                           PoolingMethod pooling_method,
                           int padding) {
    m_k_h_sz = pooling_shape[0];
    m_k_w_sz = pooling_shape[1];
    m_h_stride = pooling_shape[2];
    m_h_stride = pooling_shape[3];

    m_pooling_method = pooling_method;
    m_padding = padding;
}

void PoolingLayer::extractImagePatches(const Tensor4xf& bottom, Tensor5xf& patches) {
    patches = bottom.extract_image_patches(m_k_h_sz, m_k_w_sz, m_h_stride, m_w_stride, 1, 1, Eigen::PADDING_VALID);
}

Eigen::DSizes<int, 4> PoolingLayer::getTopShape(const Tensor4xf& bottom) {
    Eigen::DSizes<int, 4> top_shape;
    top_shape[0] = bottom.dimension(0);
    top_shape[1] = Eigen::divup(float(bottom.dimension(1) - m_k_h_sz + 1), float(m_h_stride));
    top_shape[2] = Eigen::divup(float(bottom.dimension(2) - m_k_w_sz + 1), float(m_w_stride));
    top_shape[3] = bottom.dimension(3);

    return top_shape;
}

void PoolingLayer::forward(const Tensor4xf& bottom,
                           Tensor4xf& top,
                           const Eigen::ThreadPoolDevice& device) {
    Eigen::array<int, 2> reduction_dims{1, 2};
    Eigen::DSizes<int, 4> post_reduce_diums = getTopShape(bottom);
    Tensor5xf patches; 

    extractImagePatches(bottom, patches);
    BaseFunctions::printShape(patches);
}

}
