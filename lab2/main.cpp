#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>
#include <omp.h>

using namespace std;
using namespace chrono;

// Чтение матрицы из CSV
vector<vector<double>> readMatrix(const string& filename) {
    vector<vector<double>> matrix;  
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        return matrix;
    }
    string line;
    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string val;
        while (getline(ss, val, ',')) {
            row.push_back(stod(val));
        }
        matrix.push_back(row);
    }
    file.close();
    return matrix;
}

// Запись результата в CSV – теперь с числом потоков
void writeResult(const string& filename, int size, int threads, long long time_ms) {
    ofstream out(filename, ios::app);
    if (!out.is_open()) {
        cerr << "Error: cannot open file for writing " << filename << endl;
        return;
    }
    // если файл пуст, добавить заголовок
    out.seekp(0, ios::end);
    if (out.tellp() == 0) {
        out << "size,threads,time_ms\n";
    }
    out << size << "," << threads << "," << time_ms << "\n";
    out.close();
}

int main() {
    cout << "=== Parallel Matrix Multiplication with OpenMP ===\n";

    // Запрос количества потоков у пользователя
    int num_threads;
    cout << "Enter number of threads (1, 2, 4, 8, ...): ";
    cin >> num_threads;
    cin.ignore();
    omp_set_num_threads(num_threads);

    string fileA, fileB, fileResult;
    char choice;

    do {
        // Ввод путей
        cout << "\nEnter the path to first matrix (A): ";
        getline(cin, fileA);
        cout << "Enter the path to second matrix (B): ";
        getline(cin, fileB);
        cout << "Enter the path to result CSV: ";
        getline(cin, fileResult);

        // Загрузка матриц
        vector<vector<double>> A = readMatrix(fileA);
        vector<vector<double>> B = readMatrix(fileB);

        // Проверка корректности
        if (A.empty() || B.empty()) {
            cerr << "One of the matrices is not loaded. Check the files.\n";
            continue;
        }
        int n = A.size();
        if (A[0].size() != n || B.size() != n || B[0].size() != n) {
            cerr << "Error: matrices must be square and of the same size.\n";
            continue;
        }

        // Подготовка матрицы результата
        vector<vector<double>> C(n, vector<double>(n, 0.0));

        // Замер времени
        auto start = high_resolution_clock::now();

        // Параллельное умножение с использованием OpenMP
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < n; ++k) {
                double aik = A[i][k];
                for (int j = 0; j < n; ++j) {
                    C[i][j] += aik * B[k][j];
                }
            }
        }

        auto end = high_resolution_clock::now();
        long long duration = duration_cast<milliseconds>(end - start).count();

        // Сохранение результата
        writeResult(fileResult, n, num_threads, duration);
        cout << "Multiplication completed. Size: " << n << "x" << n
             << ", threads: " << num_threads << ", time: " << duration << " ms.\n";
        cout << "Result appended to " << fileResult << "\n";

        cout << "\nDo you want to repeat with another matrix pair? (y/n): ";
        cin >> choice;
        cin.ignore();

    } while (choice == 'y' || choice == 'Y');

    cout << "Program finished.\n";
    return 0;
}
