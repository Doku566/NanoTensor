#include "nanotensor/quantization.hpp"
#include <algorithm>
#include <cmath>

namespace nanotensor {
namespace quant {

QuantizedTensor quantize_symmetric(const Tensor& input) {
    QuantizedTensor q_tensor;
    q_tensor.shape = input.shape();
    
    // 1. Find max abs value
    float max_abs = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(input[i]));
    }

    // 2. Calculate scale
    // Avoid division by zero
    if (max_abs == 0.0f) max_abs = 1.0f;
    q_tensor.scale = max_abs / 127.0f;
    
    // 3. Quantize
    q_tensor.data.resize(input.size());
    float inv_scale = 1.0f / q_tensor.scale;

    for (size_t i = 0; i < input.size(); ++i) {
        float val = input[i] * inv_scale;
        // Clamp to [-127, 127]
        val = std::max(-127.0f, std::min(127.0f, val));
        q_tensor.data[i] = static_cast<int8_t>(std::round(val));
    }
    
    return q_tensor;
}

Tensor dequantize(const QuantizedTensor& input) {
    Tensor t(input.shape);
    for (size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<float>(input.data[i]) * input.scale;
    }
    return t;
}

}
}
