import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
df = pd.read_csv('D:/Study/3 course/parallel/lab5/results.csv')

# Группировка по размеру и числу процессов
grouped = df.groupby(['size', 'processes'], as_index=False).agg(
    mean_time=('time_ms', 'mean'),
    std_time=('time_ms', 'std')
)

# Сводная таблица среднего времени (размеры по строкам, процессы по столбцам)
pivot_mean = grouped.pivot(index='size', columns='processes', values='mean_time')
pivot_std = grouped.pivot(index='size', columns='processes', values='std_time')

print("=== Среднее время выполнения (мс) ===")
print(pivot_mean.round(2))

# Ускорение относительно 1 процесса
if 1 in pivot_mean.columns:
    base = pivot_mean[1]
    speedup = pd.DataFrame(index=pivot_mean.index)
    for proc in pivot_mean.columns:
        if proc != 1:
            speedup[proc] = base / pivot_mean[proc]
    print("\n=== Ускорение (относительно 1 процесса) ===")
    print(speedup.round(2))

# Построение графика с погрешностями
plt.figure(figsize=(10, 6))
processes = sorted(grouped['processes'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(processes)))

for proc, color in zip(processes, colors):
    subset = grouped[grouped['processes'] == proc].sort_values('size')
    plt.errorbar(subset['size'], subset['mean_time'],
                 label=f'{proc} processes', marker='o', capsize=5, color=color)

plt.xlabel('Matrix size (N)', fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)
plt.title('MPI Matrix Multiplication: Time vs Size and Number of Processes', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('mpi_time_vs_size.png', dpi=300)
plt.show()