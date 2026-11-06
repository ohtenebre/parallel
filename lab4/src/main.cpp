#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

const int RUNS = 5;

struct ProfileResult
{
    double forward_time = 0.0;
    double backward_time = 0.0;
    double total_time() const { return forward_time + backward_time; }
};

void generate_system(int n, std::vector<double> &A, std::vector<double> &b, std::vector<double> &x_exact)
{
    A.resize(n * n, 0.0);
    b.assign(n, 0.0);
    x_exact.assign(n, 1.0);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    for (int i = 0; i < n; ++i)
    {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j)
        {
            A[i * n + j] = dist(gen);
            row_sum += A[i * n + j];
        }

        A[i * n + i] += row_sum;

        for (int j = 0; j < n; ++j)
            b[i] += A[i * n + j] * x_exact[j];
    }
}

double calc_max_error(const std::vector<double> &x1, const std::vector<double> &x2)
{
    double max_err = 0.0;
    for (size_t i = 0; i < x1.size(); ++i)
        max_err = std::max(max_err, std::abs(x1[i] - x2[i]));
    return max_err;
}

ProfileResult gauss_sequential(int n, std::vector<double> A, std::vector<double> b, std::vector<double> &x)
{
    ProfileResult prof;
    x.assign(n, 0.0);

    double t1 = omp_get_wtime();
    for (int i = 0; i < n - 1; ++i)
    {
        for (int k = i + 1; k < n; ++k)
        {
            double coef = A[k * n + i] / A[i * n + i];
            for (int j = i; j < n; ++j)
                A[k * n + j] -= coef * A[i * n + j];
            b[k] -= coef * b[i];
        }
    }
    prof.forward_time = omp_get_wtime() - t1;

    double t2 = omp_get_wtime();
    for (int i = n - 1; i >= 0; --i)
    {
        x[i] = b[i] / A[i * n + i];
        for (int k = 0; k < i; ++k)
            b[k] -= A[k * n + i] * x[i];
    }
    prof.backward_time = omp_get_wtime() - t2;

    return prof;
}

ProfileResult gauss_parallel(int n, std::vector<double> A, std::vector<double> b, std::vector<double> &x, int num_threads)
{
    ProfileResult prof;
    x.assign(n, 0.0);
    omp_set_num_threads(num_threads);

    double t1 = omp_get_wtime();
#pragma omp parallel
    {
        for (int i = 0; i < n - 1; ++i)
        {
#pragma omp for schedule(guided)
            for (int k = i + 1; k < n; ++k)
            {
                double coef = A[k * n + i] / A[i * n + i];
                for (int j = i; j < n; ++j)
                    A[k * n + j] -= coef * A[i * n + j];
                b[k] -= coef * b[i];
            }
        }
    }
    prof.forward_time = omp_get_wtime() - t1;

    double t2 = omp_get_wtime();
#pragma omp parallel
    {
        for (int i = n - 1; i >= 0; --i)
        {
#pragma omp single
            {
                x[i] = b[i] / A[i * n + i];
            }
#pragma omp for schedule(static)
            for (int k = 0; k < i; ++k)
                b[k] -= A[k * n + i] * x[i];
        }
    }
    prof.backward_time = omp_get_wtime() - t2;

    return prof;
}

int main()
{
    int n = 1500;
    std::vector<double> A, b, x_exact;
    generate_system(n, A, b, x_exact);

    int thread_count[] = { 1, 2, 4, 6, 8, 16, 32 };

    auto out = std::ofstream("stat.csv");
    if (!out.is_open())
        return 1;
    out << "Threads,Time_Total,Time_Forward,Time_Backward,Max_Error\n";

    for (const int &t_c : thread_count)
    {
        double log_total = 0.0;
        double log_fwd = 0.0;
        double log_bwd = 0.0;
        double last_err = 0.0;

        for (int run_c = 0; run_c < RUNS; ++run_c)
        {
            std::vector<double> x;
            ProfileResult res = (t_c == 1)
                                    ? gauss_sequential(n, A, b, x)
                                    : gauss_parallel(n, A, b, x, t_c);

            log_total += std::log(res.total_time());
            log_fwd += std::log(res.forward_time);
            log_bwd += std::log(res.backward_time);

            last_err = calc_max_error(x, x_exact);
        }

        double mean_total = std::exp(log_total / RUNS);
        double mean_fwd = std::exp(log_fwd / RUNS);
        double mean_bwd = std::exp(log_bwd / RUNS);

        out << t_c << "," << mean_total << "," << mean_fwd << "," << mean_bwd << "," << last_err << "\n";
    }

    return 0;
}
