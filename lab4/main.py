import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("stat.csv")

plt.figure(figsize=(10, 6))

plt.plot(df["Threads"], df["Time_Total"], label="Total Time", marker="o", linewidth=2)
plt.plot(
    df["Threads"], df["Time_Forward"], label="Forward Step", marker="s", linestyle="--"
)
plt.plot(
    df["Threads"], df["Time_Backward"], label="Backward Step", marker="^", linestyle=":"
)

plt.xlabel("Количество потоков")
plt.ylabel("Время (секунды)")
plt.xscale("log", base=2)
plt.xticks(df["Threads"], df["Threads"])
plt.legend()
plt.grid(True)

plt.savefig("bin/plot.png")
plt.show()
