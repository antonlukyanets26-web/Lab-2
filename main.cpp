#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <immintrin.h> // для возможной явной векторизации (опционально)
#include <omp.h>
// Подключение MKL BLAS
#include <mkl.h> // обязательно: предоставляет cblas_dgemm и управление потоками MKL

// Глобальные параметры
constexpr int N = 4096;

// Вывод информации об авторе и группе
void print_author_info() {
    // Замените на реальные данные
    const std::string author = "Лукьянец Антон Дмитриевич";
    const std::string group = "090304-РПИа-о25";
    std::cout << "Автор: " << author << "\n";
    std::cout << "Группа: " << group << "\n";
}

// Генерация матрицы NxN, элементы типа double
void generate_matrix(std::vector<double>& A, int n) {
    A.resize(n * n);
    std::mt19937_64 rng(123456); // фиксированный сид для воспроизводимости
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = dist(rng);
    }
}

// Непосредственное перемножение: C = A * B (row-major)
void multiply_naive(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Вариант 2: MKL BLAS через cblas_dgemm
void multiply_blas_mkl(const double* A, const double* B, double* C, int n) {
    // C = A * B, размерность n x n, row-major
    // cblas_dgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    // Для row-major: M = n, N = n, K = n
    const CBLAS_ORDER order = CblasRowMajor;
    const CBLAS_TRANSPOSE transA = CblasNoTrans;
    const CBLAS_TRANSPOSE transB = CblasNoTrans;
    const int M = n;
    const int N = n;
    const int K = n;
    const double alpha = 1.0;
    const double beta = 0.0;
    // lda/ldb/ldc для row-major равны размерности по строкам
    cblas_dgemm(order, transA, transB,
        M, N, K,
        alpha,
        A, n,
        B, n,
        beta,
        C, n);
}

// Оптимизированное блокированное перемножение
void multiply_tiled(const double* A, const double* B, double* C, int n, int tile = 64) {
    // tile = 64 → блок 64x64 doubles = 32 КБ, отлично помещается в L1 кэш
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += tile) {
        for (int jj = 0; jj < n; jj += tile) {
            int i_max = std::min(ii + tile, n);
            int j_max = std::min(jj + tile, n);

            for (int kk = 0; kk < n; kk += tile) {
                int k_max = std::min(kk + tile, n);

                // Порядок i -> k -> j гарантирует последовательный доступ к памяти
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        double a_val = A[i * n + k];
#pragma omp simd
                        for (int j = jj; j < j_max; ++j) {
                            C[i * n + j] += a_val * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

// Вспомогательный таймер
struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double elapsed() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        return diff.count();
    }
};

int main() {
    print_author_info();

    // Выделение памяти под матрицы
    std::vector<double> A, B;
    generate_matrix(A, N);
    generate_matrix(B, N);

    // Выходные матрицы для трёх вариантов
    std::vector<double> C_naive(N * N, 0.0);
    std::vector<double> C_blas(N * N, 0.0);
    std::vector<double> C_tiled_parallel(N * N, 0.0);

    // Отключаем динамическую балансировку потоков MKL внутри программы, чтобы воспроизводимость
    // (можно выставлять по желанию с помощью MKL_SET_NUM_THREADS())
    // Например: MKL_SET_NUM_THREADS(4);
    // Если хотите использовать все ядра по умолчанию, можно не вызывать.
    // Пример: устанавливаем ограничение в 4 потока
    // Note: закомментируйте строку, если хотите использовать все доступные потоки.
    //MKL_SET_NUM_THREADS(4);

    // 2-й вариант: MKL dgemm
    {
        std::fill(C_blas.begin(), C_blas.end(), 0.0);
        Timer timer;
        timer.start();
        multiply_blas_mkl(A.data(), B.data(), C_blas.data(), N);
        double t = timer.elapsed();
        const double c = 2.0 * N * N * N;
        const double mflops = c / t / 1e6;
        std::cout << "Variant 2 (MKL dgemm) time: " << t << " s, MFLOPS: " << mflops << "\n";
    }

    // 3-й вариант: tiled parallel
    {
        std::fill(C_tiled_parallel.begin(), C_tiled_parallel.end(), 0.0);
        Timer timer;
        timer.start();
        multiply_tiled(A.data(), B.data(), C_tiled_parallel.data(), N, 64);
        double t = timer.elapsed();
        const double c = 2.0 * N * N * N;
        const double mflops = c / t / 1e6;
        std::cout << "Variant 3 (tiling+parallel) time: " << t << " s, MFLOPS: " << mflops << "\n";
    }

    // 1-й вариант: naive
    {
        std::fill(C_naive.begin(), C_naive.end(), 0.0);
        Timer timer;
        timer.start();
        multiply_naive(A.data(), B.data(), C_naive.data(), N);
        double t = timer.elapsed();
        const double c = 2.0 * N * N * N; // число операций умножения+сложения
        const double mflops = c / t / 1e6;
        std::cout << "Variant 1 (naive) time: " << t << " s, MFLOPS: " << mflops << "\n";
    }

    // Проверка корректности: сравним MKL и naive по норме разности
    double diff = 0.0;
    for (size_t i = 0; i < C_naive.size(); ++i) diff += std::abs(C_naive[i] - C_blas[i]);
    std::cout << "L1-norm diff between naive and MKL results: " << diff << "\n";

    return 0;
}