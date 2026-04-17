%%cuda
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

using namespace std;

// Ядро умножения матриц (каждый поток – один элемент C)
__global__ void matMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Измерение времени выполнения ядра (в миллисекундах)
double measureTime(int N, int blockX, int blockY,
                   const double* d_A, const double* d_B, double* d_C) {
    dim3 block(blockX, blockY);
    dim3 grid((N + blockX - 1) / blockX, (N + blockY - 1) / blockY);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    // Параметры эксперимента
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    vector<pair<int,int>> blocks = {{8,8}, {16,16}, {32,32}};
    const int repeats = 10;               // количество повторений
    const string filename = "cuda_benchmark.csv";
    
    // Создаём CSV-файл с заголовком
    ofstream out(filename);
    if (!out) {
        cerr << "Cannot create file " << filename << endl;
        return 1;
    }
    out << "size,block_x,block_y,avg_time_ms\n";
    
    cout << "CUDA benchmark started. Repeats per config: " << repeats << endl;
    
    for (int N : sizes) {
        cout << "\nTesting size " << N << "x" << N << endl;
        
        // Генерация случайных матриц на хосте (значения 0..1000)
        vector<double> h_A(N * N), h_B(N * N);
        for (int i = 0; i < N * N; ++i) {
            h_A[i] = rand() % 1001;
            h_B[i] = rand() % 1001;
        }
        
        // Выделение памяти на GPU
        double *d_A, *d_B, *d_C;
        size_t bytes = N * N * sizeof(double);
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        
        // Копирование данных на GPU
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
        
        // Прогрев (один запуск с блоком 16x16)
        dim3 warmBlock(16,16);
        dim3 warmGrid((N+15)/16, (N+15)/16);
        matMulKernel<<<warmGrid, warmBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        
        // Тестируем каждую конфигурацию блоков
        for (auto [bx, by] : blocks) {
            if (bx * by > 1024) {
                cout << "  Skipping " << bx << "x" << by << " (too many threads per block)" << endl;
                continue;
            }
            
            vector<double> times;
            for (int r = 0; r < repeats; ++r) {
                double t = measureTime(N, bx, by, d_A, d_B, d_C);
                times.push_back(t);
            }
            
            double sum = 0.0;
            for (double t : times) sum += t;
            double avg = sum / repeats;
            
            out << N << "," << bx << "," << by << "," << avg << "\n";
            cout << "  Block " << bx << "x" << by << " -> avg " << avg << " ms" << endl;
        }
        
        // Освобождение памяти GPU
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    out.close();
    cout << "\nBenchmark finished. Results saved to " << filename << endl;
    return 0;
}