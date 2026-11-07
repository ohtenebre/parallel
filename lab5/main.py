import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("stat.csv")

plt.figure(figsize=(10, 6))

plt.plot(df["Threads"], df["Size_10000"], label="10 000 элементов", marker="o")
plt.plot(df["Threads"], df["Size_20000"], label="20 000 элементов", marker="o")
plt.plot(df["Threads"], df["Size_30000"], label="30 000 элементов", marker="o")
plt.plot(df["Threads"], df["Size_40000"], label="40 000 элементов", marker="o")

plt.xlabel("Количество потоков")
plt.ylabel("Время (миллисекунды)")
plt.xscale("log", base=2)
plt.xticks(df["Threads"], df["Threads"])
plt.legend()
plt.grid(True)

plt.savefig("bin/plot.png")
plt.show()
