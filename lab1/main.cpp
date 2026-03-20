#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>

using namespace std;
using namespace chrono;

// Чтение матрицы из CSV
vector<vector<double>> readMatrix(const string& filename) {
    vector<vector<double>> matrix;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cant open file " << filename << endl;
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

// Запись результата в CSV (добавление)
void writeResult(const string& filename, int size, long long time_ms) {
    ofstream out(filename, ios::app);
    if (!out.is_open()) {
        cerr << "Error: cant open file for reading " << filename << endl;
        return;
    }
    // если файл пуст, добавить заголовок
    out.seekp(0, ios::end);
    if (out.tellp() == 0) {
        out << "size,time_ms\n";
    }
    out << size << "," << time_ms << "\n";
    out.close();
}

int main() {
    cout << "=== Multuply square matrix ===\n";

    string fileA, fileB, fileResult;
    char choice;

    do {
        // Ввод путей
        cout << "Enter the path to first matrix (A): ";
        getline(cin, fileA);
        cout << "Enter the path to second matrix (B): ";
        getline(cin, fileB);
        cout << "Enter the path to result: ";
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
            cerr << "Error: matrix should be equal size.\n";
            continue;
        }

        // Подготовка матрицы результата
        vector<vector<double>> C(n, vector<double>(n, 0.0));

        // Замер времени
        auto start = high_resolution_clock::now();

        // Умножение
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
        writeResult(fileResult, n, duration);
        cout << "Multiply completed. Size: " << n << "x" << n << ", time: " << duration << " ms.\n";
        cout << "Result added to " << fileResult << "\n";

        cout << "\nAre you want to repeat? (y/n): ";
        cin >> choice;
        cin.ignore();

    } while (choice == 'y' || choice == 'Y');

    cout << "Program destroyed.\n";
    return 0;
}