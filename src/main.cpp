#include "nanotensor/tensor.hpp"
#include "nanotensor/ops.hpp"
#include "nanotensor/quantization.hpp"
#include <iostream>
#include <iomanip>
#include <random>

void print_tensor_stats(const std::string& name, const nanotensor::Tensor& t) {
    std::cout << "[INFO] " << name << " Shape: (";
    for(size_t s : t.shape()) std::cout << s << ",";
    std::cout << "\b) | Memory: " << (t.size() * sizeof(float)) / 1024.0f << " KB" << std::endl;
}

int main() {
    std::cout << "=== NanoTensor: From-Scratch Inference Engine ===\n" << std::endl;

    // 1. Setup Architecture params (GPT-2 Small styleish)
    size_t SeqLen = 128; // Tokens
    size_t EmbedDim = 64; // Small embedding for demo
    size_t Heads = 1;     // Single head for simplicity
    
    std::cout << "Architecting Self-Attention Layer..." << std::endl;
    std::cout << "Sequence Length: " << SeqLen << " | Embedding Dim: " << EmbedDim << std::endl;

    // 2. Initialize Q, K, V with random data
    nanotensor::Tensor Q({SeqLen, EmbedDim});
    nanotensor::Tensor K({SeqLen, EmbedDim});
    nanotensor::Tensor V({SeqLen, EmbedDim});
    nanotensor::Tensor Output({SeqLen, EmbedDim});

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for(size_t i=0; i<Q.size(); ++i) {
        Q[i] = dist(gen);
        K[i] = dist(gen);
        V[i] = dist(gen);
    }

    // 3. Run Attention (The Core)
    std::cout << "\n[CPU] Executing Scaled Dot-Product Attention..." << std::endl;
    // Timer could be added here
    nanotensor::ops::scaled_dot_product_attention(Q, K, V, Output);
    std::cout << "[SUCCESS] Attention Output Computed." << std::endl;

    print_tensor_stats("Output", Output);

    // 4. Quantization Demo
    std::cout << "\n[OPTIMIZATION] Quantizing Query Tensor (FP32 -> INT8)..." << std::endl;
    auto q_Q = nanotensor::quant::quantize_symmetric(Q);
    
    float original_mb = (Q.size() * sizeof(float)) / 1024.0f;
    float quant_mb = (q_Q.data.size() * sizeof(int8_t) + sizeof(float)) / 1024.0f; // Data + Scale
    
    std::cout << "Original Size: " << original_mb << " KB" << std::endl;
    std::cout << "Quantized Size: " << quant_mb << " KB" << std::endl;
    std::cout << "Compression Ratio: " << original_mb / quant_mb << "x" << std::endl;

    std::cout << "\n[SYSTEM] NanoTensor core modules online." << std::endl;
    return 0;
}
