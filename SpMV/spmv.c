// soal 2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h>

#define NUM_THREADS 8
#define VECTOR_SIZE 8

int **random_matrix(int m, int n, float sparsity)
{
    int **A = (int **)malloc(m * sizeof(int *));
    for (int i = 0; i < m; i++)
        A[i] = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = 0;

    for (int i = 0; i < m * n * sparsity; i++)
    {
        int v = (rand() % 10) + 1;
        int r = rand() % m;
        int c = rand() % n;

        if (A[r][c] == 0){
            A[r][c] = v;
        }
        else{
            i--;
        }   
    }

    return A;
}

int *random_vector(int length)
{
    int *v = (int *)malloc(length * sizeof(int));
    for (int i = 0; i < length; i++)
        v[i] = (rand() % 10) + 1;
    return v;
}

void compress(int **A, int *values, int *colIndex, int *rowIndex, int m, int n)
{
    int k = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (A[i][j] != 0)
            {
                values[k] = A[i][j];
                colIndex[k] = i;
                rowIndex[k] = j;
                k++;
            }
}

int *simd(int *values, int *colIndex, int *rowIndex, int *x, int m, int non_zero_count)
{
    int *ans = (int*) calloc(m, sizeof(int));
    int i, j;
    __m256i val, idx, row, x_val, ans_val;
    __m256i *vec_val = (__m256i *) values;
    __m256i *vec_col = (__m256i *) colIndex;

    for (i = 0; i < non_zero_count; i += 8) {
        val = _mm256_loadu_si256(vec_val);
        idx = _mm256_loadu_si256(vec_col);
        row = _mm256_loadu_si256((__m256i *) &rowIndex[i]);

        x_val = _mm256_i32gather_epi32(x, idx, sizeof(int));
        ans_val = _mm256_i32gather_epi32(ans, row, sizeof(int));
        ans_val = _mm256_add_epi32(ans_val, _mm256_mullo_epi32(val, x_val));
        _mm256_i32scatter_epi32(ans, row, ans_val, sizeof(int));
        
        vec_val++;
        vec_col++;
    }

    // Handle remaining elements
    for (j = i - 8; j < non_zero_count; j++)
        ans[rowIndex[j]] += (values[j] * x[colIndex[j]]);
}


int* naive_mul(int **A, int *x, int m, int n)
{
    int *ans = (int*) malloc(m * sizeof(int));

    for (int i = 0; i < m; i++){
        ans[i] = 0;
        for (int j = 0; j < n; j++)
            ans[i] += A[i][j] * x[j];
    }

    return ans;
}


int* omp_spmv(int *values, int *colIndex, int *rowIndex, int *x, int m, int non_zero_count)
{
    int *ans = (int*) calloc(m, sizeof(int));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < non_zero_count; i++)
        ans[rowIndex[i]] += (values[i] * x[colIndex[i]]);

    return ans;
}


int* spmv(int *values, int *colIndex, int *rowIndex, int *x, int m, int non_zero_count)
{
    int *ans = (int*)calloc(m, sizeof(int));

    for (int i = 0; i < non_zero_count; i++)
        ans[rowIndex[i]] += (values[i] * x[colIndex[i]]);

    return ans;
}

int main()
{
    int m = 8192;
    int n = 8192;
    float sparsity = 0.125;

    int **A = random_matrix(m, n, sparsity);
    int *x = random_vector(n);

    int non_zero_count = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (A[i][j] != 0)
                non_zero_count++;

    int *values = (int *)malloc(non_zero_count * sizeof(int));
    int *colIndex = (int *)malloc(non_zero_count * sizeof(int));
    int *rowIndex = (int *)malloc(non_zero_count * sizeof(int));
    
    compress(A, values, colIndex, rowIndex, m, n);

    double start, end;

    start = omp_get_wtime();
    int* ans_naive = naive_mul(A, x, m, n);
    end = omp_get_wtime();
    printf("naive mul time: %f\n", (end - start) * 1000);

    start = omp_get_wtime();
    int* ans_spmv = spmv(values, colIndex, rowIndex, x, m, non_zero_count);
    end = omp_get_wtime();
    printf("spmv time: %f\n", (end - start) * 1000);

    start = omp_get_wtime();
    int* ans_omp = omp_spmv(values, colIndex, rowIndex, x, m, non_zero_count);
    end = omp_get_wtime();
    printf("omp time: %f\n", (end - start) * 1000);


    start = omp_get_wtime();
    int* ans_simd = simd(values, colIndex, rowIndex, x, m, non_zero_count);
    end = omp_get_wtime();
    printf("simd time: %f\n", (end - start) * 1000);

    return 0;
}
