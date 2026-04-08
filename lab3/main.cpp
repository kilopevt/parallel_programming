#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>

using namespace std;

// Чтение матрицы из CSV (только для процесса 0)
vector<vector<double>> readMatrix(const string& fname) {
    vector<vector<double>> M;
    ifstream f(fname);
    if (!f.is_open()) return M;
    string line;
    while (getline(f, line)) {
        vector<double> row;
        size_t pos = 0;
        while (pos < line.size()) {
            size_t comma = line.find(',', pos);
            if (comma == string::npos) comma = line.size();
            row.push_back(stod(line.substr(pos, comma - pos)));
            pos = comma + 1;
        }
        M.push_back(row);
    }
    return M;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) 
            cerr << "Usage: mpiexec -n N " << argv[0] << " A.csv B.csv results.csv\n";
        MPI_Finalize();
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string outFile = argv[3];

    int n = 0;
    vector<double> A, B;

    // Процесс 0 читает матрицы
    if (rank == 0) {
        auto MA = readMatrix(fileA);
        auto MB = readMatrix(fileB);
        if (MA.empty() || MB.empty() || MA.size() != MA[0].size() || MA.size() != MB.size()) {
            cerr << "Invalid matrices\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = MA.size();
        A.resize(n*n);
        B.resize(n*n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                A[i*n + j] = MA[i][j];
                B[i*n + j] = MB[i][j];
            }
    }

    // Рассылаем размер
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    // Все процессы получают матрицы
    if (rank != 0) {
        A.resize(n*n);
        B.resize(n*n);
    }
    MPI_Bcast(A.data(), n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределение строк
    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    vector<double> C(local_rows * n, 0.0);

    // Синхронизация и замер времени
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Умножение (только свою часть)
    for (int i = 0; i < local_rows; i++) {
        int gi = start_row + i;
        for (int k = 0; k < n; k++) {
            double aik = A[gi * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += aik * B[k * n + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = (t1 - t0) * 1000.0; // мс

    // Собираем максимальное время (самый медленный процесс)
    double global_time;
    MPI_Reduce(&elapsed, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Процесс 0 записывает результат
    if (rank == 0) {
        ofstream fout(outFile, ios::app);
        if (fout.tellp() == 0)
            fout << "size,processes,time_ms\n";
        fout << n << "," << size << "," << global_time << "\n";
        fout.close();
        cout << "Size: " << n << "x" << n << ", processes: " << size
             << ", time: " << global_time << " ms\n";
    }

    MPI_Finalize();
    return 0;
}