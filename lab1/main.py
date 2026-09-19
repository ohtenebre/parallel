import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("bin/stat.csv")

plt.figure(figsize=(7, 4))
plt.plot(df["Threads"], df["Time"], marker="o", color="blue", linewidth=2)
plt.title("Время работы (Среднее геометрическое из 100 прогонов)")
plt.xlabel("Потоки")
plt.ylabel("Время (наносекунды)")
plt.xscale("log", base=2)
plt.xticks(df["Threads"], df["Threads"])
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("bin/plot_time.png")
plt.close()
