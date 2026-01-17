#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <memory>
#include <cstring>
#include <cmath>

namespace nanotensor {

/**
 * @brief Minimal Tensor class for educational purposes.
 * Manages raw float memory without smart pointers (to allow manual memory manipulation understanding).
 * 
 * Supports: 
 * - Contiguous memory layout
 * - Basic float32 storage
 */
class Tensor {
public:
    Tensor(const std::vector<size_t>& shape) : shape_(shape) {
        size_t total_elements = 1;
        for (auto s : shape_) total_elements *= s;
        
        // Manual allocation to show we understand memory
        data_ = new float[total_elements];
        size_ = total_elements;
        
        // Initialize to 0
        std::memset(data_, 0, size_ * sizeof(float));
    }

    // Destructor
    ~Tensor() {
        if (data_) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    // Rule of Five: Delete Copy to prevent double-free (unique ownership semantics for raw ptr demo)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move constructor
    Tensor(Tensor&& other) noexcept : data_(other.data_), shape_(std::move(other.shape_)), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] data_; // Clean up current resources
            
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            size_ = other.size_;
            
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    float* data() { return data_; }
    const float* data() const { return data_; }
    
    size_t size() const { return size_; }
    const std::vector<size_t>& shape() const { return shape_; }
    
    // Accessor (Linear)
    float& operator[](size_t idx) { return data_[idx]; }
    const float& operator[](size_t idx) const { return data_[idx]; }

    // Helper for 2D access
    float& at(size_t r, size_t c) {
        // Assume last two dims are Rows/Cols
        size_t cols = shape_.back();
        return data_[r * cols + c]; 
    }

private:
    float* data_;
    std::vector<size_t> shape_;
    size_t size_;
};

} // namespace nanotensor
