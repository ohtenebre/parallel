import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("bin/stat.csv")

plt.figure(figsize=(9, 5))

plt.plot(df["Threads"], df["Time_Rows"], marker="o", color="blue", linewidth=2, label="По строкам (dot_rows)")
plt.plot(df["Threads"], df["Time_Cols"], marker="s", color="red", linewidth=2, label="По столбцам (dot_cols)")
plt.plot(df["Threads"], df["Time_Blocks"], marker="^", color="green", linewidth=2, label="Блочный 64x64 (dot_blocks)")

plt.title("Сравнение эффективности алгоритмов умножения матрицы на вектор", fontsize=12, fontweight='bold')
plt.xlabel("Количество потоков", fontsize=10)
plt.ylabel("Время работы (миллисекунды)", fontsize=10)

plt.xscale("log", base=2)
plt.xticks(df["Threads"], df["Threads"])

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig("bin/block.png", dpi=300)
plt.close()
