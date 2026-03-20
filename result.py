import pandas as pd
import matplotlib.pyplot as plt
import os

# Чтение результатов
results_file = "result/result.csv"
if not os.path.exists(results_file):
    print(f"Файл {results_file} не найден. Запустите сначала C++ программу.")
    exit(1)

df = pd.read_csv(results_file)

# Сортировка по размеру для красивого графика
df = df.sort_values("size")

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(df["size"], df["time_ms"], marker='o', linestyle='-', color='b')
plt.xlabel("Размер матрицы (N)")
plt.ylabel("Время выполнения (мс)")
plt.title("Зависимость времени умножения матриц от размера")
plt.grid(True)
plt.savefig("time_vs_size.png", dpi=300)
plt.show()

print("График сохранён как time_vs_size.png")