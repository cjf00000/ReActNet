#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>

int sqr(int x) { return x * x; }

// Greedy algorithm
std::pair<int, int> calc_quant_bin(torch::Tensor hist) {
    TORCH_CHECK(hist.device().is_cpu(), "hist must be a CPU tensor!");
    TORCH_CHECK(hist.is_contiguous(), "hist must be contiguous!");
    auto *h = hist.data_ptr<float>();

    int N = hist.size(0);

    // e[i, j] = \sum_k=0^j h[k](k-i)^2
    std::vector<std::vector<double>> e(N);
    for (int i = 0; i < N; i++) {
        e[i].resize(N);
        double sum = 0;
        for (int j = 0; j < N; j++)
            e[i][j] = sum += h[j] * sqr(j - i);
    }

    int best_mid = -1;
    int best_delta = -1;
    double best_diff = 1e20;
    for (int mid = 0; mid < N; mid++)
        for (int delta = 1; delta < std::min(mid, N - 1 - mid); delta++) {
            int left = mid - delta;
            int right = mid + delta;
            // mid is counted as left
            float obj = e[left][mid] + e[right][N-1] - e[right][mid];
            if (obj < best_diff) {
                best_diff = obj;
                best_mid = mid;
                best_delta = delta;
            }
        }

    int left = best_mid - best_delta;
    int right = best_mid + best_delta;

    // Rationality check
    double diff = 0;
    for (int i = 0; i <= best_mid; i++)
        diff += h[i] * sqr(i - left);
    for (int i = best_mid + 1; i < N; i++)
        diff += h[i] * sqr(i - right);

    std::cout << "Best diff " << best_diff << ' ' << diff << std::endl;
    return std::make_pair(left, right);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("calc_quant_bin", &calc_quant_bin, "calc_quant_bin");
}
