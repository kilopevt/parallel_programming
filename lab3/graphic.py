import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Папка с результатами
result_dir = "result"

# Регулярное выражение для извлечения числа процессов и размера из имени файла
# Поддерживаются оба варианта: result_C2_400.csv и results_C2_400.csv
pattern = re.compile(r"results?[_-]C(\d+)[_-](\d+)\.csv", re.IGNORECASE)

# Словарь для хранения списков времени: ключ = (процессы, размер)
data = {}

# Проход по всем файлам в папке result
for filename in os.listdir(result_dir):
    match = pattern.match(filename)
    if not match:
        continue
    procs = int(match.group(1))
    size = int(match.group(2))
    filepath = os.path.join(result_dir, filename)

    # Чтение CSV-файла с заголовком
    df_file = pd.read_csv(filepath)
    # Предполагаем, что столбец со временем называется 'time_ms'
    # Если название другое, можно автоматически найти числовой столбец
    if 'time_ms' in df_file.columns:
        times = df_file['time_ms'].tolist()
    else:
        # Если столбец называется иначе, берём первый числовой столбец
        numeric_cols = df_file.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            times = df_file[numeric_cols[0]].tolist()
        else:
            print(f"Warning: no numeric column in {filename}, skipping")
            continue

    # Добавляем все времена из этого файла в общий список для данной комбинации
    key = (procs, size)
    if key not in data:
        data[key] = []
    data[key].extend(times)

# Преобразуем в DataFrame для удобства
rows = []
for (procs, size), times in data.items():
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time)**2 for t in times) / (len(times) - 1))**0.5 if len(times) > 1 else 0.0
    rows.append({
        "processes": procs,
        "size": size,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
        "count": len(times)
    })

df = pd.DataFrame(rows)
df = df.sort_values(["processes", "size"])

# Построение графика
plt.figure(figsize=(10, 6))
processes_list = sorted(df["processes"].unique())
colors = plt.cm.tab10(range(len(processes_list)))

for procs, color in zip(processes_list, colors):
    subset = df[df["processes"] == procs].sort_values("size")
    sizes = subset["size"].values
    means = subset["mean_time_ms"].values
    stds = subset["std_time_ms"].values
    plt.errorbar(sizes, means, yerr=stds, label=f"{procs} processes", 
                 marker='o', capsize=5, color=color, linewidth=2)

plt.xlabel("Matrix size (N)", fontsize=12)
plt.ylabel("Average time (ms)", fontsize=12)
plt.title("MPI Matrix Multiplication: Time vs Size and Number of Processes", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("mpi_time_vs_size.png", dpi=300)
plt.show()

# Сводная таблица среднего времени
pivot_time = df.pivot(index="size", columns="processes", values="mean_time_ms")
print("\n=== Average execution time (ms) ===")
print(pivot_time.round(1))

# Таблица ускорения относительно 1 процесса
if 1 in pivot_time.columns:
    base = pivot_time[1]
    speedup = pd.DataFrame(index=pivot_time.index)
    for procs in pivot_time.columns:
        if procs != 1:
            speedup[procs] = base / pivot_time[procs]
    print("\n=== Speedup (relative to 1 process) ===")
    print(speedup.round(2))
else:
    print("\nWarning: no data for 1 process, speedup not calculated.")

# Дополнительно выведем количество замеров для каждой комбинации
pivot_count = df.pivot(index="size", columns="processes", values="count")
print("\n=== Number of measurements per combination ===")
print(pivot_count)