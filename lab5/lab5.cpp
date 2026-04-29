#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Проверка аргументов: ожидается один параметр - размер матрицы
    if (argc != 2) {
        if (rank == 0) {
            cerr << "Usage: mpiexec -n N " << argv[0] << " <matrix_size>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        if (rank == 0) cerr << "Matrix size must be positive.\n";
        MPI_Finalize();
        return 1;
    }

    // Генерация матриц только на процессе 0
    vector<double> A, B;
    if (rank == 0) {
        // Инициализируем генератор случайных чисел
        srand(time(nullptr));
        A.resize(n * n);
        B.resize(n * n);
        for (int i = 0; i < n * n; ++i) {
            A[i] = rand() % 1001;     // целые от 0 до 1000
            B[i] = rand() % 1001;
        }
    }

    // Рассылаем размер всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Все процессы получают матрицы
    if (rank != 0) {
        A.resize(n * n);
        B.resize(n * n);
    }
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределение строк матрицы C между процессами
    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    vector<double> C(local_rows * n, 0.0);

    // Барьер для синхронизации перед замером времени
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Параллельное умножение: каждый процесс вычисляет свои строки
    for (int i = 0; i < local_rows; ++i) {
        int gi = start_row + i;               // глобальный индекс строки
        for (int k = 0; k < n; ++k) {
            double aik = A[gi * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed_ms = (t1 - t0) * 1000.0;

    // Собираем максимальное время выполнения среди всех процессов
    double global_time_ms;
    MPI_Reduce(&elapsed_ms, &global_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Процесс 0 записывает результат в CSV-файл
    if (rank == 0) {
        ofstream fout("results.csv", ios::app);
        if (fout.tellp() == 0) {   // если файл пуст, пишем заголовок
            fout << "size,processes,time_ms\n";
        }
        fout << n << "," << size << "," << global_time_ms << "\n";
        fout.close();
        cout << "Multiplication completed. Size: " << n << "x" << n
             << ", processes: " << size << ", time: " << global_time_ms << " ms\n";
    }

    MPI_Finalize();
    return 0;
}