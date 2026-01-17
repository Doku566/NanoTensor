#pragma once
#include "tensor.hpp"

namespace nanotensor {
namespace quant {

    struct QuantizedTensor {
        std::vector<int8_t> data;
        float scale;
        std::vector<size_t> shape;
    };

    /**
     * @brief Quantizes a float tensor to int8 symmetric.
     * scale = max(abs(x)) / 127.0
     * q = x / scale
     */
    QuantizedTensor quantize_symmetric(const Tensor& input);

    /**
     * @brief Dequantizes int8 to float for verification.
     * x = q * scale
     */
    Tensor dequantize(const QuantizedTensor& input);

}
}
