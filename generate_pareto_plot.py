import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data
comparison_data = [
    {"Model": "Qwen2-VL-7B (Full VLM)", "Accuracy": 0.9600, "Latency_p95_ms": 2400.0, "Params_M": 7000.0, "PEAS": 0.0212},
    {"Model": "InternVideo2-B (Video MAE)", "Accuracy": 0.9400, "Latency_p95_ms": 450.0, "Params_M": 300.0, "PEAS": 0.0399},
    {"Model": "VideoMAE V2 (Action Head)", "Accuracy": 0.8800, "Latency_p95_ms": 280.0, "Params_M": 86.0, "PEAS": 0.0483},
    {"Model": "Ours: Kinetic Gated-Attention MIL", "Accuracy": 0.7391, "Latency_p95_ms": 1.32, "Params_M": 0.351, "PEAS": 3.327}
]

# Sort by latency for Pareto line
sorted_pts = sorted(comparison_data, key=lambda x: x["Latency_p95_ms"])
x_pts = [p["Latency_p95_ms"] for p in sorted_pts]
y_pts = [p["Accuracy"] for p in sorted_pts]
peas = [p["PEAS"] for p in sorted_pts]
models = [p["Model"] for p in sorted_pts]

# Bubble size mapping: clear minimum size so all models are visible, but 'Ours' is prominently larger
# PEAS range: 0.02 to 5.17
bubble_sizes = [130 + (p["PEAS"] / 5.1721) * 550 for p in sorted_pts]

# Plot setup
fig, ax = plt.subplots(figsize=(10.5, 6), dpi=150)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#fafbfc')

# Pareto frontier curve (connecting non-dominated trade-off points)
ax.plot(x_pts, y_pts, linestyle='--', color='#34495e', linewidth=2.2, alpha=0.75, zorder=2, label='Pareto Frontier (Empirical)')

# Subtle shaded area under frontier showing efficiency envelope
# Create a smooth interpolation in log-space for fill
log_x = np.log10(x_pts)
fine_log_x = np.linspace(log_x[0], log_x[-1], 200)
fine_x = 10 ** fine_log_x
fine_y = np.interp(fine_log_x, log_x, y_pts)
ax.fill_between(fine_x, 0.50, fine_y, color='#3498db', alpha=0.04, zorder=1)

# Scatter bubbles
scatter = ax.scatter(
    x_pts, y_pts,
    s=bubble_sizes,
    c=peas,
    cmap='viridis',
    edgecolors='#1c2833',
    linewidth=1.8,
    alpha=0.92,
    zorder=4
)

# Colorbar with clean styling
cbar = fig.colorbar(scatter, ax=ax, pad=0.03, fraction=0.038, aspect=24)
cbar.set_label('Novel PEAS Score (Higher = Superior Pareto Efficiency)', fontsize=9.5, weight='bold', color='#2c3e50', labelpad=10)
cbar.ax.tick_params(labelsize=9)

# Clean, collision-free annotations for each point
# 1. Ours: Kinetic Gated-Attention MIL (x=1.32, y=0.7391)
ax.annotate(
    "★ Ours: Kinetic Gated-Attention MIL\n"
    "• Latency p95: 1.32 ms (1,800× faster)\n"
    "• Accuracy: 73.91% (0.806 Macro-F1)\n"
    "• Footprint: 351K params (1.3 MB)\n"
    "• PEAS Score: 3.327 (160× higher)",
    xy=(1.32, 0.7391),
    xytext=(2.8, 0.73),
    fontsize=9.2,
    weight='semibold',
    color='#1a252f',
    bbox=dict(boxstyle="round,pad=0.55", facecolor="#fff9e6", edgecolor="#f39c12", linewidth=1.6, alpha=0.96),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.08", color="#e67e22", lw=1.6),
    zorder=5
)

# 2. VideoMAE V2 (Action Head) (x=280, y=0.88)
ax.annotate(
    "VideoMAE V2 (Action Head)\n"
    "86M params | 280 ms | PEAS: 0.048",
    xy=(280.0, 0.88),
    xytext=(75.0, 0.865),
    fontsize=8.8,
    weight='bold',
    color='#2c3e50',
    ha='right',
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1.2, alpha=0.92),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="#7f8c8d", lw=1.2),
    zorder=5
)

# 3. InternVideo2-B (Video MAE) (x=450, y=0.94)
ax.annotate(
    "InternVideo2-B (Video MAE)\n"
    "300M params | 450 ms | PEAS: 0.040",
    xy=(450.0, 0.94),
    xytext=(140.0, 0.96),
    fontsize=8.8,
    weight='bold',
    color='#2c3e50',
    ha='right',
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1.2, alpha=0.92),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", color="#7f8c8d", lw=1.2),
    zorder=5
)

# 4. Qwen2-VL-7B (Full VLM) (x=2400, y=0.96)
# Placed to the left so it NEVER overlaps with the colorbar
ax.annotate(
    "Qwen2-VL-7B (Full VLM)\n"
    "7.0B params | 2400 ms | PEAS: 0.021",
    xy=(2400.0, 0.96),
    xytext=(1300.0, 0.925),
    fontsize=8.8,
    weight='bold',
    color='#2c3e50',
    ha='right',
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#bdc3c7", linewidth=1.2, alpha=0.92),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="#7f8c8d", lw=1.2),
    zorder=5
)

# Direction of Optimization Badge in top-left
ax.annotate(
    "▲ Higher Accuracy\n◀ Lower Latency (Faster)\n[Optimal Frontier Direction]",
    xy=(0.025, 0.96),
    xycoords='axes fraction',
    fontsize=8.5,
    weight='bold',
    color='#196f3d',
    va='top',
    ha='left',
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#eafaf1", edgecolor="#a9dfbf", linewidth=1.2, alpha=0.95)
)

# Axes, Scales, and Limits
ax.set_xscale('log')
ax.set_xlim(0.35, 6000.0)
ax.set_ylim(0.53, 1.02)

# Axis labels & Titles
ax.set_xlabel("Latency p95 (ms) [Log Scale]", fontsize=10.5, weight='bold', color='#2c3e50', labelpad=8)
ax.set_ylabel("Top-1 Classification Accuracy", fontsize=10.5, weight='bold', color='#2c3e50', labelpad=8)
ax.set_title("Pareto Frontier: Accuracy vs. Latency vs. Model Footprint", fontsize=12.5, weight='bold', color='#1a252f', pad=14)

# Gridlines
ax.grid(True, which="major", ls="-", color="#e5e7e9", alpha=0.9, zorder=0)
ax.grid(True, which="minor", ls=":", color="#f2f4f4", alpha=0.7, zorder=0)

# Clean tick formatting
ax.tick_params(axis='both', which='major', labelsize=9.5, colors='#2c3e50')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

# Subtitle / note
ax.text(
    0.5, 1.015,
    "Bubble size and color represent PEAS score  |  Dashed line shows the empirical Pareto efficiency frontier",
    transform=ax.transAxes,
    fontsize=9,
    color='#566573',
    ha='center',
    va='bottom'
)

plt.tight_layout()
output_png = r"c:\Desktop\Table\Job\task_1_video_classifier\pareto_frontier_well.png"
fig.savefig(output_png, format="png", dpi=200, bbox_inches="tight")
print(f"Saved figure to {output_png}")
