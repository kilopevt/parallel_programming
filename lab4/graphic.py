import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cuda_benchmark.csv')

# График зависимости времени от размера для разных конфигураций блоков
plt.figure(figsize=(10,6))
for (bx, by), group in df.groupby(['block_x','block_y']):
    label = f'block {bx}x{by}'
    plt.plot(group['size'], group['avg_time_ms'], marker='o', label=label)

plt.xlabel('Matrix size (N)')
plt.ylabel('Average time (ms)')
plt.title('CUDA Matrix Multiplication Performance')
plt.grid(True)
plt.legend()
plt.show()

# Таблица средних значений
pivot = df.pivot(index='size', columns=['block_x','block_y'], values='avg_time_ms')
print(pivot.round(2))