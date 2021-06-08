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
    auto *h = hist.data_ptr<int>();

    int N = hist.size(0);

    // e[i, j] = \sum_k=0^j h[k](k-i)^2
    std::vector<std::vector<float>> e(N);
    for (int i = 0; i < N; i++) {
        e[i].resize(N);
        int sum = 0;
        for (int j = 0; j < N; j++)
            e[i][j] = sum += h[j] * sqr(j - i);
    }

    int best_mid = -1;
    int best_delta = -1;
    float best_diff = -1;
    for (int mid = 0; )

    // min \sum_i C_i / (2^b_i - 1)^2, s.t., \sum_i b_i = N b
    std::priority_queue<std::pair<float, int>> q;


    auto *C_data = C.data_ptr<float>();
    auto *w_data = w.data_ptr<int>();

    auto get_obj = [&](float C, int b) {
        int coeff_1 = ((1 << b) - 1) * ((1 << b) - 1);
        int coeff_2 = ((1 << (b-1)) - 1) * ((1 << (b-1)) - 1);
        return C * (1.0 / coeff_1 - 1.0 / coeff_2);     // negative
    };

    int N = b.size(0);
    double b_sum = 0;
    for (int i = 0; i < N; i++) {
        auto delta = get_obj(C_data[i], b_data[i]) / w_data[i];
        q.push(std::make_pair(delta, i));
        b_sum += b_data[i] * w_data[i];
    }

    while (b_sum > target) {        // Pick up the smallest increment (largest decrement)
        assert(!q.empty());
        auto i = q.top().second;
        q.pop();
        b_data[i] -= 1;
        b_sum -= w_data[i];
        if (b_data[i] > 1) {
            auto delta = get_obj(C_data[i], b_data[i]) / w_data[i];
            q.push(std::make_pair(delta, i));
        }
    }
    return b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("calc_quant_bin", &calc_quant_bin, "calc_quant_bin");
}
