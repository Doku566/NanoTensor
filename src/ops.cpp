#include "nanotensor/ops.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace nanotensor {
namespace ops {

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    // A: [M, K], B: [K, N] -> C: [M, N]
    // Simplified for 2D.
    size_t M = A.shape()[0];
    size_t K = A.shape()[1];
    size_t N = B.shape()[1];

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                // Accessing raw data assuming row-major
                // A[i, k] * B[k, j]
                size_t idx_A = i * K + k;
                size_t idx_B = k * N + j;
                sum += A.data()[idx_A] * B.data()[idx_B];
            }
            C.data()[i * N + j] = sum;
        }
    }
}

void softmax(Tensor& x) {
    // Softmax on the last dimension
    size_t rows = x.shape()[0];
    size_t cols = x.shape()[1];

    for (size_t i = 0; i < rows; ++i) {
        // 1. Find max for stability
        float max_val = -1e9;
        for (size_t j = 0; j < cols; ++j) {
            max_val = std::max(max_val, x.data()[i * cols + j]);
        }

        // 2. Exponentiate and sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float val = std::exp(x.data()[i * cols + j] - max_val);
            x.data()[i * cols + j] = val;
            sum_exp += val;
        }

        // 3. Normalize
        for (size_t j = 0; j < cols; ++j) {
            x.data()[i * cols + j] /= sum_exp;
        }
    }
}

void scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& output) {
    // Simplified: Assume 2D [Seq, Dim] for single head for demo
    // Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(dk) ) * V
    
    size_t Seq = Q.shape()[0];
    size_t Dim = Q.shape()[1];
    
    // 1. Scores = Q * K^T
    // We need K Transposed.
    // Instead of transposing physically, we adjust matmul indices.
    
    // Temp buffer for scores [Seq, Seq]
    Tensor scores({Seq, Seq});
    float scale = 1.0f / std::sqrt(static_cast<float>(Dim));

    for (size_t i = 0; i < Seq; ++i) { // Query idx
        for (size_t j = 0; j < Seq; ++j) { // Key idx
            float dot = 0.0f;
            for (size_t d = 0; d < Dim; ++d) {
                // Q[i, d] * K[j, d] (Note: K is accessed as [j, d], which is effectively K^T if we dot product rows)
                // Wait, mathematically Q * K^T. 
                // Row i of Q dot Col j of K^T -> Row i of Q dot Row j of K. Yes.
                
                dot += Q.data()[i * Dim + d] * K.data()[j * Dim + d];
            }
            scores.data()[i * Seq + j] = dot * scale;
        }
    }

    // 2. Softmax(Scores)
    softmax(scores);

    // 3. Output = Scores * V
    // [Seq, Seq] * [Seq, Dim] -> [Seq, Dim]
    matmul(scores, V, output);
}

} // namespace ops
} // namespace nanotensor
