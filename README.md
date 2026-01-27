# NanoTensor: From-Scratch Transformer Engine

**NanoTensor** is a minimalist C++ inference engine designed to deconstruct the Transformer architecture. It implements the critical `Scaled Dot-Product Attention` mechanism without relying on high-level frameworks like PyTorch or TensorFlow, exposing the raw mathematical operations and memory management challenges of LLMs.

## Core Architecture: The Attention Mechanism

The engine manually computes the attention scores.
Formula:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Implementation Details
*   **Q (Query), K (Key), V (Value)**: These are projections of the input tokens.
*   **$O(N^2)$ Complexity**: The term $QK^T$ results in a $[SeqLen \times SeqLen]$ matrix.
    *   For sequence length $N=128$, this is trivial.
    *   For $N=100k$, this matrix consumes gigabytes of RAM. This implementation reveals why naive Transformers scale quadratically with sequence length.

## Optimization: Quantization
NanoTensor includes an INT8 quantization module (`src/quantization.cpp`) to simulate the compression used in production models (like LLaMA.cpp).
*   **Method**: Symmetric Linear Quantization.
*   **Effect**: Reduces memory footprint by ~75% (4 bytes $\to$ 1 byte per weight).
*   **Storage**: One FP32 scale factor is stored per tensor (Tensor-wise quantization).

## The "Scar": Why this is slow
Running this on a CPU without fused kernels (FlashAttention) or BLAS libraries (OpenBLAS/MKL) highlights the importance of hardware-aware optimization.
*   **Bottleneck**: The nested loops in `ops::matmul` cause massive cache thrashing.
*   **Missing KV Cache**: This implementation re-computes $K$ and $V$ for previous tokens at every step (in a generation scenario), illustrating the redundancy that KV Caching solves.

## Project Structure
*   `include/nanotensor/tensor.hpp`: Manual memory management (Raw pointers).
*   `src/ops.cpp`: Mathematical implementation of Attention.
*   `src/quantization.cpp`: FP32 to INT8 conversion logic.

## WebAssembly Demo (Browser Inference)
To compile NanoTensor for the web (WASM), use Emscripten:

```bash
emcc src/ops.cpp src/wasm_bindings.cpp -I include \
  -o web/nanotensor.js \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap']" \
  --bind \
  -O3
```
Then serve the `web/` folder (`python -m http.server`) and open `index.html`.
