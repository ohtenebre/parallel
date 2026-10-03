#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

std::vector<float> dot_rows(const std::vector<std::vector<float>> &matrix, const std::vector<float> &vec)
{
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    if (cols != vec.size())
        return {};

    std::vector<float> res(rows, 0.f);

#pragma omp parallel for
    for (size_t i = 0; i < rows; ++i)
    {
        int thread_id = omp_get_thread_num();

        float sum = 0.f;
        for (int j = 0; j < cols; ++j)
        {
            sum += matrix[i][j] * vec[j];
        }
        res[i] = sum;
    }

    return res;
}

std::vector<float> dot_cols(const std::vector<std::vector<float>> &matrix, const std::vector<float> &vec)
{
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    if (cols != vec.size())
        return {};

    std::vector<float> res(rows, 0.f);
    float *res_ptr = res.data();

#pragma omp parallel for reduction(+ : res_ptr[0 : rows])
    for (size_t i = 0; i < cols; ++i)
    {
        for (int j = 0; j < rows; ++j)
            res_ptr[j] += matrix[j][i] * vec[i];
    }

    return res;
}

std::vector<float> dot_blocks(const std::vector<std::vector<float>> &matrix, const std::vector<float> &vec, size_t block_size)
{
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    if (cols != vec.size())
        return {};

    std::vector<float> res(rows, 0.f);

#pragma omp parallel for
    for (size_t bi = 0; bi < rows; bi += block_size)
        for (size_t bj = 0; bj < cols; bj += block_size)
            for (size_t i = bi; i < std::min(bi + block_size, rows); ++i)
                for (size_t j = bj; j < std::min(bj + block_size, cols); ++j)
                    res[i] += matrix[i][j] * vec[j];

    return res;
}

int main()
{
    int rows = 40000;
    int cols = 40000;

    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols, 0.f));
    std::vector<float> vec(cols, 0.f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            matrix[i][j] = dis(gen);

    for (size_t i = 0; i < cols; ++i)
        vec[i] = dis(gen);

    int thread_count[] = { 1, 2, 4, 6, 8, 12, 16, 32, 64 };

    auto out = std::ofstream("bin/stat.csv");
    if (!out.is_open())
        return 1;

    out << "Threads,Time_Rows,Time_Cols,Time_Blocks\n";

    for (const int &t_c : thread_count)
    {
        omp_set_num_threads(t_c);

        auto start_r = std::chrono::steady_clock::now();
        auto res_r = dot_rows(matrix, vec);
        auto end_r = std::chrono::steady_clock::now();
        auto dur_r = std::chrono::duration_cast<std::chrono::milliseconds>(end_r - start_r);

        auto start_c = std::chrono::steady_clock::now();
        auto res_c = dot_cols(matrix, vec);
        auto end_c = std::chrono::steady_clock::now();
        auto dur_c = std::chrono::duration_cast<std::chrono::milliseconds>(end_c - start_c);

        auto start_b = std::chrono::steady_clock::now();
        auto res_b = dot_blocks(matrix, vec, 16);
        auto end_b = std::chrono::steady_clock::now();
        auto dur_b = std::chrono::duration_cast<std::chrono::milliseconds>(end_b - start_b);

        out << t_c << "," << dur_r.count() << "," << dur_c.count() << "," << dur_b.count() << "\n";
    }

    return 0;
}
