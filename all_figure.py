import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator

# 폰트 및 글로벌 스타일 설정
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 0.8

# ==========================================================
# 1. Data & Style Definitions
# ==========================================================

steps = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

ordered_models = [
    "SD1.5", "SD2.1", "SDXL", "SD3.5-Medium", "SD3.5-Large", "SD3.5-Turbo", "FLUX.1"
]

styles = {
    'SD1.5':        {'color': '#ff7f0e', 'marker': 'o'},
    'SD2.1':        {'color': '#e377c2', 'marker': 's'},
    'SD3.5-Medium': {'color': '#17becf', 'marker': '^'},
    'SD3.5-Turbo':  {'color': '#9467bd', 'marker': 'd'},
    'SDXL':         {'color': '#2ca02c', 'marker': 'P'},
    'SD3.5-Large':  {'color': '#1f77b4', 'marker': 'v'},
    'FLUX.1':       {'color': '#d62728', 'marker': '*'}
}

# --- Data ---
latency_data = {
    'SD1.5':        [0.130, 0.188, 0.237, 0.286, 0.336, 0.385, 0.434, 0.486, 0.534, 0.776, 1.026, 1.259],
    'SD2.1':        [0.328, 0.429, 0.538, 0.643, 0.755, 0.866, 0.978, 1.081, 1.198, 1.751, 2.304, 2.864],
    'SDXL':         [2.926, 4.219, 5.550, 6.802, 8.159, 9.381, 10.719, 12.042, 13.245, 19.921, 26.468, 33.007],
    'SD3.5-Medium': [1.409, 1.982, 2.579, 3.160, 3.747, 4.345, 4.879, 5.514, 6.090, 9.030, 11.822, 14.716],
    'SD3.5-Large':  [3.389, 4.984, 6.599, 8.177, 9.788, 11.343, 12.963, 14.532, 16.158, 24.106, 34.888, 40.123],
    'SD3.5-Turbo':  [2.125, 3.065, 4.021, 4.974, 5.913, 6.877, 7.811, 8.679, 9.582, 14.381, 18.871, 23.524],
    'FLUX.1':       [4.291, 6.256, 8.181, 10.216, 11.986, 13.965, 16.031, 18.036, 20.327, 29.972, 39.663, 49.375]
}

clip_data = {
    'SD1.5':        [0.290, 0.309, 0.311, 0.312, 0.313, 0.314, 0.313, 0.314, 0.315, 0.314, 0.314, 0.313],
    'SD2.1':        [0.281, 0.299, 0.306, 0.309, 0.310, 0.309, 0.311, 0.311, 0.311, 0.312, 0.312, 0.312],
    'SDXL':         [0.254, 0.302, 0.314, 0.317, 0.318, 0.318, 0.319, 0.319, 0.320, 0.319, 0.319, 0.318],
    'SD3.5-Medium': [0.244, 0.285, 0.305, 0.316, 0.319, 0.321, 0.321, 0.321, 0.322, 0.323, 0.323, 0.322],
    'SD3.5-Large':  [0.269, 0.304, 0.314, 0.319, 0.320, 0.321, 0.322, 0.322, 0.323, 0.322, 0.323, 0.322],
    'SD3.5-Turbo':  [0.319, 0.318, 0.317, 0.317, 0.317, 0.317, 0.316, 0.316, 0.316, 0.316, 0.316, 0.316],
    'FLUX.1':       [0.302, 0.311, 0.313, 0.313, 0.313, 0.312, 0.312, 0.312, 0.312, 0.311, 0.310, 0.310]
}

lpips_data = {
    'SD1.5':        [0.653, 0.582, 0.528, 0.496, 0.464, 0.432, 0.411, 0.393, 0.350, 0.306, 0.217, 0.169],
    'SD2.1':        [0.634, 0.559, 0.501, 0.456, 0.425, 0.400, 0.380, 0.364, 0.322, 0.281, 0.191, 0.148],
    'SDXL':         [0.680, 0.626, 0.579, 0.541, 0.509, 0.480, 0.455, 0.433, 0.392, 0.330, 0.237, 0.189],
    'SD3.5-Medium': [0.718, 0.681, 0.647, 0.616, 0.583, 0.558, 0.537, 0.517, 0.499, 0.425, 0.360, 0.306],
    'SD3.5-Large':  [0.750, 0.729, 0.705, 0.682, 0.656, 0.633, 0.610, 0.589, 0.568, 0.484, 0.422, 0.375],
    'SD3.5-Turbo':  [0.653, 0.590, 0.550, 0.520, 0.498, 0.479, 0.462, 0.446, 0.431, 0.372, 0.325, 0.283],
    'FLUX.1':       [0.631, 0.593, 0.558, 0.536, 0.514, 0.494, 0.475, 0.458, 0.439, 0.369, 0.312, 0.275]
}

pickscore_data = {
    'SD1.5':        [19.699, 20.562, 20.927, 21.104, 21.195, 21.276, 21.310, 21.349, 21.381, 21.438, 21.460, 21.472],
    'SD2.1':        [19.965, 20.813, 21.193, 21.389, 21.481, 21.557, 21.605, 21.639, 21.691, 21.739, 21.789, 21.792],
    'SDXL':         [17.424, 18.011, 18.572, 18.917, 19.149, 19.324, 19.440, 19.530, 19.711, 19.816, 19.995, 20.046],
    'SD3.5-Medium': [18.926, 20.433, 21.231, 21.702, 22.016, 22.196, 22.296, 22.354, 22.403, 22.486, 22.512, 22.509],
    'SD3.5-Large':  [18.675, 19.819, 20.542, 21.007, 21.283, 21.515, 21.694, 21.795, 21.876, 22.009, 22.011, 21.994],
    'SD3.5-Turbo':  [18.522, 18.998, 19.284, 19.572, 19.752, 19.967, 20.145, 20.309, 20.464, 21.088, 21.384, 21.510],
    'FLUX.1':       [21.283, 22.431, 22.735, 22.896, 22.943, 22.977, 22.984, 23.007, 23.000, 23.006, 22.981, 22.974]
}

imagereward_data = {
    'SD1.5':        [-1.146, -0.381, -0.141, -0.001, 0.061, 0.112, 0.114, 0.151, 0.151, 0.185, 0.199, 0.207],
    'SD2.1':        [-0.894, -0.171,  0.085,  0.209, 0.257, 0.297, 0.332, 0.357, 0.369, 0.395, 0.416, 0.422],
    'SDXL':         [-2.215, -2.030, -1.747, -1.539,-1.377,-1.271,-1.179,-1.093,-0.976,-0.887,-0.745,-0.697],
    'SD3.5-Medium': [-1.627, -0.348, -0.348,  0.595, 0.767, 0.879, 0.919, 0.955, 0.981, 1.021, 1.022, 1.048],
    'SD3.5-Large':  [-1.771, -0.821, -0.225,  0.179, 0.344, 0.515, 0.619, 0.694, 0.719, 0.790, 0.800, 0.805],
    'SD3.5-Turbo':  [-1.667, -1.394, -1.170, -1.049,-0.881,-0.695,-0.532,-0.399,-0.250, 0.260, 0.509, 0.617],
    'FLUX.1':       [ 0.159,  0.725,  0.833,  0.891, 0.911, 0.929, 0.940, 0.950, 0.958, 0.956, 0.936, 0.939]
}

# ==========================================================
# 2. Grid Layout Setup
# ==========================================================

fig = plt.figure(figsize=(7.0, 3.2))

gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1.0, 2.1], wspace=0.15)

ax_lat = fig.add_subplot(gs_outer[0])

gs_right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[1], wspace=0.18, hspace=0.32)

ax_clip  = fig.add_subplot(gs_right[0, 0])
ax_lpips = fig.add_subplot(gs_right[0, 1])
ax_pick  = fig.add_subplot(gs_right[1, 0])
ax_img   = fig.add_subplot(gs_right[1, 1])

subplots = [
    (ax_lat, latency_data, "Latency (s)", (-2, 53), [0, 10, 20, 30, 40, 50]),
    (ax_clip, clip_data, "CLIP", (0.235, 0.335), [0.24, 0.26, 0.28, 0.30, 0.32]),
    (ax_lpips, lpips_data, "LPIPS", (0.15, 0.78), [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    (ax_pick, pickscore_data, "PickScore", (17.5, 23.5), [18, 19, 20, 21, 22, 23]),
    (ax_img, imagereward_data, "ImageReward", (-2.3, 1.2), [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
]

x_major_ticks = [4, 8, 12, 16, 20, 30, 40, 50]
x_minor_ticks = [6, 10, 14, 18]

# ==========================================================
# 3. Drawing Subplots
# ==========================================================

for ax, data_dict, ylabel, ylim, yticks in subplots:
    for model in ordered_models:
        ax.plot(
            steps, data_dict[model],
            color=styles[model]['color'],
            marker=styles[model]['marker'],
            markersize=2.8,
            linewidth=0.7,
            markerfacecolor='white',
            markeredgewidth=0.7,
            label=model
        )
    
    ax.set_xlabel("Time Step", fontsize=7, labelpad=3.0)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=2.0)
    
    ax.set_xticks(x_major_ticks)
    ax.xaxis.set_minor_locator(FixedLocator(x_minor_ticks))
    ax.set_yticks(yticks)
    ax.set_xlim(2, 52)
    ax.set_ylim(ylim)
    
    ax.tick_params(
        direction="in",
        which="major",
        labelsize=6,
        length=3.0,
        pad=2,
        top=False,
        right=False
    )
    ax.tick_params(
        direction="in",
        which="minor",
        length=3.0,
        top=False,
        right=False
    )
    ax.grid(False)

# ==========================================================
# 4. Top Legend Layout (동적 정밀 위치 지정)
# ==========================================================

plt.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.98)

# 실제 그릴 좌표 위치 계산을 위한 draw
fig.canvas.draw()

# 왼쪽 서브플롯 축과 오른쪽 서브플롯 축 위치 파악
pos_left = ax_lat.get_position()
pos_right = ax_lpips.get_position()

x_start = pos_left.x0
x_width = pos_right.x1 - pos_left.x0

handles, labels = ax_lat.get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc="lower left",
    bbox_to_anchor=(x_start, 0.89, x_width, 0.08),
    mode="expand",
    ncol=7,
    fontsize=6.5,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    handletextpad=0.2,
    borderpad=0.3
)

# 저장
plt.savefig("figure_perfect_align.pdf", bbox_inches="tight")
plt.savefig("figure_perfect_align.png", dpi=300, bbox_inches="tight")
