import matplotlib.pyplot as plt
import numpy as np

# 数据准备
bar_data = {
    'A': [1.916, 2.025, 2.208, 2.161, 1.591, 1.544, 1.244, 1.272, 1.211, 1.143, 1.307, 1.097, 1.119, 1.055, 1.020, 1.150, 1.025, 1.045, 1.102, 1.056],
    'B': [1.418, 1.472, 1.359, 1.238, 1.051, 1.145, 1.068, 1.025, 1.002, 0.981, 1.032, 1.008, 1.008, 0.981, 0.975, 0.973, 0.980, 0.989, 1.002, 1.000],
    'C': [1.414, 1.257, 1.241, 1.102, 1.030, 1.118, 1.044, 1.011, 0.999, 0.980, 1.018, 0.996, 0.999, 0.979, 0.975, 0.966, 0.974, 0.985, 1.000, 0.999],
    'D': [1.334, 1.230, 1.226, 1.094, 1.028, 1.088, 1.029, 1.009, 0.998, 0.980, 1.003, 0.988, 0.998, 0.979, 0.975, 0.959, 0.970, 0.984, 1.000, 0.999],
    'E': [1.305, 1.220, 1.223, 1.092, 1.028, 1.076, 1.026, 1.009, 0.998, 0.980, 0.997, 0.986, 0.997, 0.978, 0.975, 0.955, 0.969, 0.984, 1.000, 0.999],
    'A100': [1] * 20
}

line_data = {
    'A': [2.091, 2.188, 2.384, 2.487, 2.132, 1.444, 1.172, 1.153, 1.205, 1.119, 1.231, 0.986, 0.952, 1.171, 1.072, 1.042, 0.889, 0.852, 1.088, 1.023],
    'B': [1.373, 1.191, 1.047, 1.029, 1.097, 1.036, 0.761, 0.637, 0.595, 0.771, 0.893, 0.673, 0.585, 0.544, 0.739, 0.756, 0.619, 0.551, 0.591, 0.761],
    'C': [1.011, 0.959, 0.822, 0.795, 0.785, 0.814, 0.628, 0.559, 0.537, 0.601, 0.710, 0.587, 0.537, 0.514, 0.618, 0.630, 0.559, 0.521, 0.570, 0.644],
    'D': [0.923, 0.866, 0.800, 0.780, 0.772, 0.736, 0.608, 0.555, 0.534, 0.598, 0.662, 0.572, 0.535, 0.513, 0.617, 0.616, 0.551, 0.524, 0.569, 0.644],
    'E': [0.930, 0.831, 0.789, 0.773, 0.768, 0.751, 0.607, 0.552, 0.534, 0.597, 0.698, 0.576, 0.534, 0.514, 0.617, 0.661, 0.556, 0.526, 0.569, 0.644],
    'A100': [1] * 20
}

labels = list(bar_data.keys())
x = np.arange(len(bar_data['A']))
bar_width = 0.1
group_size = 5
num_groups = len(x) // group_size

# 绘图
fig, ax1 = plt.subplots(figsize=(12, 5))

# 柱状图（左Y轴）
for i, label in enumerate(labels):
    offset = (i - len(labels) / 2) * bar_width + bar_width / 2
    ax1.bar(x + offset, bar_data[label], width=bar_width, label=label)

ax1.set_ylabel("Relative Throughput")
ax1.set_ylim(0, 2.5)

# 虚线分隔每组
for g in range(1, num_groups):
    pos = g * group_size - 0.5
    ax1.axvline(pos, color='gray', linestyle='--', linewidth=1)
    ax1.text(pos - 2.5, -0.1, f"Group Size: {128 * (2 ** (g - 1))}", ha='center', va='top', transform=ax1.get_xaxis_transform())

# 设置X轴
batch_sizes = ['1', '4', '16', '64', '256'] * num_groups
ax1.set_xticks(x)
ax1.set_xticklabels(batch_sizes)
ax1.set_xlabel("Batch Size per Group", labelpad=5)

# 折线图（右Y轴）
ax2 = ax1.twinx()
colors = ['blue', 'darkred', 'green', 'purple', 'orange', 'black']
for i, label in enumerate(labels):
    ax2.plot(x, line_data[label], marker='o', label=f'{label} (line)', color=colors[i], linestyle='-', linewidth=1)

ax2.set_ylabel("Relative Energy")
ax2.set_ylim(0, 2.6)

# 图例合并
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("fid.jpg")