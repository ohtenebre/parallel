import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("bin/stat.csv")

plt.figure(figsize=(10, 6))

plt.plot(df["Threads"], df["Time_lent"], label="Lent (Linear)", marker="o")

plt.plot(df["Threads"], df["Time_blocks"], label="Block (Tiling)", marker="o")

plt.xlabel("Количество потоков")
plt.ylabel("Время (микросекунды)")
plt.xscale("log", base=2)
plt.xticks(df["Threads"], df["Threads"])
plt.legend()
plt.grid(True)

plt.savefig("bin/plot.png")
plt.show()
