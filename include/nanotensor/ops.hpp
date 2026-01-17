#pragma once
#include "tensor.hpp"

namespace nanotensor {
namespace ops {

    /**
     * @brief Matrix Multiplication: C = A * B.
     * Naive implementation O(N^3).
     */
    void matmul(const Tensor& A, const Tensor& B, Tensor& C);

    /**
     * @brief Softmax along the last dimension.
     * Incorporates numerical stability (subtract max).
     */
    void softmax(Tensor& x);

    /**
     * @brief Scaled Dot-Product Attention.
     * Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
     * 
     * @param Q Query Tensor [Seq, Head, Dim]
     * @param K Key Tensor [Seq, Head, Dim]
     * @param V Value Tensor [Seq, Head, Dim]
     * @param output Result Tensor
     */
    void scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& output);

} // namespace ops
} // namespace nanotensor
