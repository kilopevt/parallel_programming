import numpy as np
import os

sizes = [200, 400, 800, 1200, 1600, 2000]

os.makedirs("matrices", exist_ok=True)

for size in sizes:
    # Генерация целых случайных чисел от 0 до 1000
    A = np.random.randint(0, 1001, size=(size, size))
    B = np.random.randint(0, 1001, size=(size, size))

    # Сохранение матриц в csv формат
    np.savetxt(f"matrices/A_{size}.csv", A, delimiter=",", fmt="%d")
    np.savetxt(f"matrices/B_{size}.csv", B, delimiter=",", fmt="%d")

    print(f"Сгенерированы матрицы размера {size}")