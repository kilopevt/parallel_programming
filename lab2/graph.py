import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Папка с CSV-файлами
data_dir = "./result"

# Паттерн для извлечения числа потоков и размера из имени файла
pattern = re.compile(r"C(\d+)_(\d+)\.csv")

# Словарь для хранения результатов: ключ = (threads, size), значение = список времён
results = {}

# Проход по всем файлам в директории
for filename in os.listdir(data_dir):
    match = pattern.match(filename)
    if not match:
        continue
    threads = int(match.group(1))
    size = int(match.group(2))
    
    filepath = os.path.join(data_dir, filename)
    # Чтение всех чисел из файла
    with open(filepath, 'r') as f:
        content = f.read()
    # Извлекаем все числа с плавающей точкой
    numbers = re.findall(r"[-+]?\d*\.?\d+", content)
    # Преобразуем в float, берём первые 3
    times = [float(x) for x in numbers[:3]]
    results[(threads, size)] = times

# Преобразование в DataFrame для удобства
data = []
for (threads, size), times in results.items():
    mean_time = np.mean(times)
    std_time = np.std(times, ddof=1) if len(times) > 1 else 0
    data.append({
        "threads": threads,
        "size": size,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "times": times
    })

df = pd.DataFrame(data)

# Сортировка по размеру и потокам
df = df.sort_values(["threads", "size"])

# Построение графика
plt.figure(figsize=(10, 6))

# Уникальные количества потоков
threads_list = sorted(df["threads"].unique())
colors = plt.cm.tab10(range(len(threads_list)))  # разные цвета для разных потоков

for threads, color in zip(threads_list, colors):
    subset = df[df["threads"] == threads].sort_values("size")
    sizes = subset["size"].values
    means = subset["mean_time_ms"].values
    stds = subset["std_time_ms"].values
    
    plt.errorbar(sizes, means, label=f"{threads} поток(ов)", 
                 capsize=5, color=color, linewidth=2)

plt.xlabel("Размер матрицы (N)", fontsize=12)
plt.ylabel("Среднее время выполнения (мс)", fontsize=12)
plt.title("Зависимость времени умножения матриц от размера и числа потоков", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("matrix_multiplication_benchmark.png", dpi=300)
plt.show()

# Вывод таблицы средних значений
pivot_table = df.pivot(index="size", columns="threads", values="mean_time_ms")
print("\n=== Среднее время выполнения (мс) ===")
print(pivot_table.round(1))

# Дополнительно: таблица с ускорением относительно 1 потока
if 1 in threads_list:
    base = pivot_table[1]
    speedup = pd.DataFrame(index=pivot_table.index)
    for t in threads_list:
        if t != 1:
            speedup[t] = base / pivot_table[t]
    print("\n=== Ускорение (относительно 1 потока) ===")
    print(speedup.round(2))