#ifndef EMBEDDING_UTILS_H
#define EMBEDDING_UTILS_H

#include <vector>
#include <cmath>
#include <numeric>
#include <opencv2/core.hpp>

inline float l2Distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e6f;
    return std::sqrt(std::inner_product(a.begin(), a.end(), b.begin(), 0.0f,
                                        std::plus<float>(),
                                        [](float x, float y) { return (x - y) * (x - y); }));
}

inline float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 0.0f;
    float dot = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    float norm1 = std::sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0f));
    float norm2 = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0f));
    return (norm1 > 0 && norm2 > 0) ? dot / (norm1 * norm2) : 0.0f;
}

inline bool isMatching(const std::vector<float>& a, const std::vector<float>& b, float threshold = 0.5f) {
    return l2Distance(a, b) < threshold;
}

#endif // EMBEDDING_UTILS_H
