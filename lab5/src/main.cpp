#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

const int RUNS = 50;

void quick_sort_seq(std::vector<double> &arr)
{
    std::sort(arr.begin(), arr.end());
}

void psrs_sort(std::vector<double> &arr, int num_threads)
{
    int n = arr.size();

    if (num_threads == 1)
    {
        std::sort(arr.begin(), arr.end());
        return;
    }

    int p = num_threads;
    int chunk = n / p;

    std::vector<double> samples(p * p);
    std::vector<double> splitters(p - 1);
    std::vector<std::vector<int>> bounds(p, std::vector<int>(p + 1));
    std::vector<double> result(n);
    std::vector<int> class_sizes(p, 0);
    std::vector<int> class_starts(p, 0);

#pragma omp parallel num_threads(p)
    {
        int tid = omp_get_thread_num();
        int start = tid * chunk;
        int end = (tid == p - 1) ? n : (tid + 1) * chunk;
        int local_n = end - start;

        std::sort(arr.begin() + start, arr.begin() + end);

        int step = local_n / p;
        for (int i = 0; i < p; ++i)
            samples[tid * p + i] = arr[start + i * step];

#pragma omp barrier

#pragma omp single
        {
            std::sort(samples.begin(), samples.end());
            for (int i = 0; i < p - 1; ++i)
                splitters[i] = samples[(i + 1) * p];
        }

        bounds[tid][0] = start;
        bounds[tid][p] = end;
        for (int i = 0; i < p - 1; ++i)
        {
            auto it = std::upper_bound(arr.begin() + start, arr.begin() + end, splitters[i]);
            bounds[tid][i + 1] = std::distance(arr.begin(), it);
        }

#pragma omp barrier

#pragma omp single
        {
            for (int i = 0; i < p; ++i)
                for (int t = 0; t < p; ++t)
                    class_sizes[i] += bounds[t][i + 1] - bounds[t][i];
            for (int i = 1; i < p; ++i)
                class_starts[i] = class_starts[i - 1] + class_sizes[i - 1];
        }

        int write_pos = class_starts[tid];
        for (int t = 0; t < p; ++t)
        {
            int part_start = bounds[t][tid];
            int part_end = bounds[t][tid + 1];
            int part_size = part_end - part_start;
            std::copy(arr.begin() + part_start, arr.begin() + part_end, result.begin() + write_pos);
            write_pos += part_size;
        }

        std::sort(result.begin() + class_starts[tid], result.begin() + class_starts[tid] + class_sizes[tid]);
    }

    arr = std::move(result);
}

bool check_sorted(const std::vector<double> &arr)
{
    for (size_t i = 1; i < arr.size(); ++i)
        if (arr[i - 1] > arr[i])
            return false;
    return true;
}

int main()
{
    std::vector<int> sizes = { 10000, 20000, 30000, 40000 };
    std::vector<int> threads = { 1, 2, 4, 6, 8, 16, 32 };

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 10000.0);

    std::vector<std::vector<double>> originals(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        originals[i].resize(sizes[i]);
        for (int j = 0; j < sizes[i]; ++j)
            originals[i][j] = dist(gen);
    }

    auto out = std::ofstream("stat.csv");
    if (!out.is_open())
        return 1;

    out << "Threads";
    for (int n : sizes)
        out << ",Size_" << n;
    out << "\n";

    for (int p : threads)
    {
        out << p;
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            double total_time = 0.0;
            bool is_correct = true;

            for (int r = 0; r < RUNS; ++r)
            {
                std::vector<double> arr = originals[i];

                auto start = std::chrono::steady_clock::now();
                psrs_sort(arr, p);
                auto end = std::chrono::steady_clock::now();

                total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                if (r == 0 && !check_sorted(arr))
                    is_correct = false;
            }

            if (!is_correct)
                std::cerr << "[err] size: " << sizes[i] << " t_c: " << p << "!\n";

            double mean_time_ms = (total_time / RUNS) / 1000.0;
            out << "," << mean_time_ms;
        }
        out << "\n";
    }

    return 0;
}
