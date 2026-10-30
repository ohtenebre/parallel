#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <span>
#include <vector>

const int RUNS = 10;
constexpr double eps = 1e-6;

// lent multithread matrix dot
std::vector<double> dot_lent(std::span<const double> m1, std::span<const double> m2, size_t m, size_t n, size_t l, int len_k)
{
    std::vector<double> result(m * n, 0.0);

#pragma omp parallel for schedule(static, len_k)
    for (size_t i = 0; i < m; ++i)
        for (size_t k = 0; k < l; ++k)
            for (size_t j = 0; j < n; ++j)
                result[i * n + j] += m1[i * l + k] * m2[k * n + j];

    return result;
}

// block multithread matrix dot
std::vector<double> dot_block(std::span<const double> m1, std::span<const double> m2, size_t m, size_t n, size_t l, size_t block_size)
{
    std::vector<double> result(m * n, 0.0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i += block_size)
    {
        for (size_t j = 0; j < n; j += block_size)
        {
            size_t i_end = std::min(i + block_size, m);
            size_t j_end = std::min(j + block_size, n);

            for (size_t k = 0; k < l; k += block_size)
            {
                size_t k_end = std::min(k + block_size, l);

                for (size_t ii = i; ii < i_end; ++ii)
                {
                    for (size_t kk = k; kk < k_end; ++kk)
                    {
                        double val = m1[ii * l + kk];

                        for (size_t jj = j; jj < j_end; ++jj)
                            result[ii * n + jj] += val * m2[kk * n + jj];
                    }
                }
            }
        }
    }
    return result;
}

int main()
{
    // Init
    size_t m = 1500;
    size_t l = 1500;
    size_t n = 1500;

    int len_k = 128;
    int block_size = 64;

    std::vector<double> m1(m * l, 0.0);
    std::vector<double> m2(l * n, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < l; ++j)
            m1[i * l + j] = dis(gen);

    for (size_t i = 0; i < l; ++i)
        for (size_t j = 0; j < n; ++j)
            m2[i * n + j] = dis(gen);

    // Run
    int thread_count[] = { 1, 2, 4, 6, 8, 12, 16, 32 };

    auto out = std::ofstream("bin/stat.csv");
    if (!out.is_open())
        return 1;
    out << "Threads,Time_lent,Time_blocks\n";

    for (const int &t_c : thread_count)
    {
        double log_time_sum1 = 0.0;
        double log_time_sum2 = 0.0;
        for (size_t run_c = 0; run_c < RUNS; ++run_c)
        {
            omp_set_num_threads(t_c);

            auto start_r = std::chrono::steady_clock::now();
            auto res_r = dot_lent(std::span(m1), std::span(m2), m, n, l, len_k);
            auto end_r = std::chrono::steady_clock::now();
            auto dur_r = std::chrono::duration_cast<std::chrono::microseconds>(end_r - start_r);

            auto start_c = std::chrono::steady_clock::now();
            auto res_c = dot_block(std::span(m1), std::span(m2), m, n, l, block_size);
            auto end_c = std::chrono::steady_clock::now();
            auto dur_c = std::chrono::duration_cast<std::chrono::microseconds>(end_c - start_c);

            log_time_sum1 += std::log(static_cast<double>(dur_r.count()));
            log_time_sum2 += std::log(static_cast<double>(dur_c.count()));
        }

        double mean1 = std::exp(log_time_sum1 / RUNS);
        double mean2 = std::exp(log_time_sum2 / RUNS);

        out << t_c << "," << mean1 << "," << mean2 << "\n";
    }

    return 0;
}
