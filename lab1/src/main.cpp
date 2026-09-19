#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <omp.h>

constexpr double eps = 1e-6;
const int RUNS = 200;
const double a = 0.4;
const double b = 1.5;

double f(double x)
{
    return (x * x * x * x) / (0.5 * x * x + x + 6);
}

double simpson(double a, double b, int n, int t_c)
{
    double len = b - a;
    double step = len / n;
    double sum = f(a) + f(b);

    omp_set_num_threads(t_c);
#pragma omp parallel for reduction(+ : sum)
    for (int i = 1; i < n; ++i)
    {
        if (i % 2 == 0)
            sum += 2.0 * f(a + i * step);
        else
            sum += 4.0 * f(a + i * step);
    }

    return sum * step / 3.0;
}

int main()
{
    auto out = std::ofstream("bin/stat.csv");
    if (!out.is_open())
        return 1;

    out << "Threads,Time,Res\n";
    int thread_count[] = { 1, 2, 4, 8, 16, 32, 64, 128 };

    for (const int &t_c : thread_count)
    {
        double log_time_sum = 0;
        double I_old = 0;

        int n = 2;
        for (int run_count = 0; run_count < RUNS; ++run_count)
        {
            n = 2000000;
            auto err = 1.0;
            auto I_new = simpson(a, b, n, t_c);

            auto start = std::chrono::steady_clock::now();
            while (err > eps)
            {
                auto I_old = I_new;
                n *= 2;
                I_new = simpson(a, b, n, t_c);
                err = std::abs(I_new - I_old) / 15.0;
            }
            auto end = std::chrono::steady_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            log_time_sum += std::log(static_cast<double>(duration.count()));
            I_old = I_new;
        }

        std::cout << n << std::endl;

        double geo_mean_time = std::exp(log_time_sum / RUNS);

        out << t_c << "," << geo_mean_time << "," << I_old << "\n";
    }

    out.close();
    return 0;
}
