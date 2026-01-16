import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def natural_sort_key(s):
    """
    自然排序的键函数
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]


def animate_point_cloud_folder(folder_path, file_prefix):
    # 获取文件夹中符合前缀的点云数据文件
    file_list = [f for f in os.listdir(folder_path) if f.startswith(file_prefix) and f.endswith('.xlsx')]
    file_list.sort(key=natural_sort_key)  # 使用自然排序对文件名进行排序

    # 创建画布和子图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    sc = ax.scatter([], [], [], c=[], cmap='rainbow')  # 使用rainbow颜色映射

    # 创建图例占位符
    legend = None

    def update(frame):
        nonlocal legend  # 使用nonlocal声明以便在函数内部修改外部的legend变量

        # 读取点云数据文件
        file_path = os.path.join(folder_path, file_list[frame])
        df = pd.read_excel(file_path)

        # 提取数据列
        x = df['X']   # 对 X 坐标进行平移
        y = df['Y']   # 对 Y 坐标进行平移
        z = df['Z']   # 对 Z 坐标进行平移
        snr = df['Noise']  # 提取 SNR 列数据

        # 更新散点图数据点位置和颜色
        sc._offsets3d = (x, y, z)
        sc.set_array(snr)

        # 更新标题
        ax.set_title(file_list[frame])

    # 设置坐标轴范围
    ax.set_xlim(2, 8)  # X 轴范围为 0 到 15
    ax.set_ylim(0.5, 2)  # Y 轴范围为 0 到 10
    ax.set_zlim(2, 8)  # Z 轴范围为 0 到 15

    # 设置坐标系网格线的样式
    ax.grid(True, linewidth=2)

    # 创建动画
    num_frames = len(file_list)
    animation = FuncAnimation(fig, update, frames=num_frames, interval=100)

    # 显示动画
    plt.show()


# 播放点云数据文件夹
folder_path = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\image\SNR\pHistBytes_clustered'
file_prefix = 'pHistBytes_'
animate_point_cloud_folder(folder_path, file_prefix)