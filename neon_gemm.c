# gcc -o neon_gemm neon_gemm.c -march=armv8-a+simd -O3
# Code generated with chatgpt

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

#define N 512  // Matrix size (N x N)

// Function to get the current time in seconds
double get_time() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// Standard (Scalar) GEMM: C = A * B
void scalar_gemm(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// NEON SIMD GEMM
void neon_gemm(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += 4) {  // Process 4 elements at a time
            float32x4_t sum_vec = vdupq_n_f32(0.0f);  // Initialize sum vector

            for (int k = 0; k < n; k++) {
                float32x4_t b_vec = vld1q_f32(&B[k * n + j]);  // Load 4 elements from B
                float32x4_t a_scalar = vdupq_n_f32(A[i * n + k]);  // Broadcast A[i, k]
                sum_vec = vmlaq_f32(sum_vec, a_scalar, b_vec);  // Multiply-accumulate
            }

            vst1q_f32(&C[i * n + j], sum_vec);  // Store result
        }
    }
}

int main() {
    float *A = (float *)aligned_alloc(16, N * N * sizeof(float));
    float *B = (float *)aligned_alloc(16, N * N * sizeof(float));
    float *C_scalar = (float *)aligned_alloc(16, N * N * sizeof(float));
    float *C_neon = (float *)aligned_alloc(16, N * N * sizeof(float));

    if (!A || !B || !C_scalar || !C_neon) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(rand() % 10);
        B[i] = (float)(rand() % 10);
        C_scalar[i] = C_neon[i] = 0.0f;
    }

    // Measure execution time for scalar GEMM
    double start_time_scalar = get_time();
    scalar_gemm(A, B, C_scalar, N);
    double end_time_scalar = get_time();

    // Measure execution time for NEON GEMM
    double start_time_neon = get_time();
    neon_gemm(A, B, C_neon, N);
    double end_time_neon = get_time();

    // Print execution times
    printf("Scalar GEMM Time: %f seconds\n", end_time_scalar - start_time_scalar);
    printf("NEON SIMD GEMM Time: %f seconds\n", end_time_neon - start_time_neon);

    // Verify results (check a few values)
    printf("First 5 results (Scalar): [%f, %f, %f, %f, %f]\n",
           C_scalar[0], C_scalar[1], C_scalar[2], C_scalar[3], C_scalar[4]);
    printf("First 5 results (NEON): [%f, %f, %f, %f, %f]\n",
           C_neon[0], C_neon[1], C_neon[2], C_neon[3], C_neon[4]);

    // Free memory
    free(A);
    free(B);
    free(C_scalar);
    free(C_neon);

    return 0;
}
