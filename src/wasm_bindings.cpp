#include <emscripten/bind.h>
#include "nanotensor/tensor.hpp"
#include "nanotensor/ops.hpp"
#include <vector>
#include <string>

using namespace emscripten;
using namespace nanotensor;

// Wrapper function for JS interaction
// Accepts flattened std::vector from JS, runs attention, returns result
std::vector<float> run_attention_wasm(int seq_len, int embed_dim, 
                                      std::vector<float> q_data, 
                                      std::vector<float> k_data, 
                                      std::vector<float> v_data) {
    
    // 1. Reconstruct Tensors
    Tensor Q({(size_t)seq_len, (size_t)embed_dim});
    Tensor K({(size_t)seq_len, (size_t)embed_dim});
    Tensor V({(size_t)seq_len, (size_t)embed_dim});
    Tensor Output({(size_t)seq_len, (size_t)embed_dim});

    // Copy data (simple loop for safety in wasm environment)
    for(size_t i=0; i<Q.size(); ++i) Q[i] = q_data[i];
    for(size_t i=0; i<K.size(); ++i) K[i] = k_data[i];
    for(size_t i=0; i<V.size(); ++i) V[i] = v_data[i];

    // 2. Run Engine
    ops::scaled_dot_product_attention(Q, K, V, Output);

    // 3. Return serialized
    std::vector<float> result;
    result.resize(Output.size());
    for(size_t i=0; i<Output.size(); ++i) result[i] = Output[i];
    
    return result;
}

EMSCRIPTEN_BINDINGS(nanotensor_module) {
    function("run_attention", &run_attention_wasm);
    register_vector<float>("VectorFloat");
}
